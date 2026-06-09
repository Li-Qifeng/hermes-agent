"""Discord server introspection and management tool (REST API + bot token).

The model-visible schema is filtered by two gates: privileged intents from GET /applications/@me
(search_members / member_info need GUILD_MEMBERS; fetch_messages / list_pins are annotated when
MESSAGE_CONTENT is missing) and the ``discord.server_actions`` config allowlist. Per-guild
permissions are NOT pre-checked — a call-time 403 is mapped to guidance by :func:`_enrich_403`.
"""

import functools
import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.secret_scope import get_secret
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
_DISCORD_RESPONSE_BODY_MAX_BYTES = 4 * 1024 * 1024
_DISCORD_ERROR_BODY_MAX_BYTES = 64 * 1024

# Application flag bits (GET /applications/@me → "flags"); the *_LIMITED bit is the
# <100-guild variant of the same intent.
_FLAGS_GUILD_MEMBERS = (1 << 14) | (1 << 15)
_FLAGS_MESSAGE_CONTENT = (1 << 18) | (1 << 19)


class DiscordAPIError(Exception):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Discord API error {status}: {body}")


def _read_limited_response_body(source: Any, limit: int, *, label: str) -> bytes:
    body = source.read(limit + 1)
    if len(body) > limit:
        raise DiscordAPIError(502, f"Discord API {label} exceeded {limit} bytes.")
    return body


def _get_bot_token() -> Optional[str]:
    """Resolve the Discord bot token under the active profile secret scope."""
    return (get_secret("DISCORD_BOT_TOKEN", "") or "").strip() or None


def _discord_request(
    method: str, path: str, token: str, params: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None, timeout: int = 15,
    reason: str = "") -> Any:
    """Make a request to the Discord REST API."""
    url = f"{DISCORD_API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {
        "Authorization": f"Bot {token}", "Content-Type": "application/json",
        "User-Agent": "Hermes-Agent (https://github.com/NousResearch/hermes-agent)"}
    if reason:
        headers["X-Audit-Log-Reason"] = urllib.parse.quote(reason)
    req = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode("utf-8"),
        method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 204:
                return None
            body = _read_limited_response_body(resp, _DISCORD_RESPONSE_BODY_MAX_BYTES, label="response body")
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            error_body = _read_limited_response_body(
                e, _DISCORD_ERROR_BODY_MAX_BYTES, label="error body").decode("utf-8", errors="replace")
        except DiscordAPIError as too_large:
            error_body = too_large.body
        except Exception:
            error_body = ""
        raise DiscordAPIError(e.code, error_body) from e


_CHANNEL_TYPE_NAMES = {
    0: "text", 2: "voice", 4: "category", 5: "announcement", 10: "announcement_thread",
    11: "public_thread", 12: "private_thread", 13: "stage", 15: "forum", 16: "media"}


def _channel_type_name(type_id: int) -> str:
    return _CHANNEL_TYPE_NAMES.get(type_id, f"unknown({type_id})")


# ── capability detection (application intents) ──────────────────────────────
# Per-token in-process cache: the app/me endpoint is hit at most once per process.
_capability_cache: Dict[str, Dict[str, Any]] = {}

# Privileged intents change only when the user flips them in the Developer Portal, so
# 24h disk staleness is harmless: a hidden action re-appears on the next refresh; an
# exposed action the bot lost fails at call time with an enriched 403.
_CAPABILITY_DISK_TTL_SECONDS = 24 * 3600

# One background detection per (process, token) at most.
_capability_bg_started: set = set()
_capability_bg_lock = threading.Lock()

# Permissive default (``detected`` False = detection failed/pending): all actions
# exposed, call-time 403s mapped to guidance by ``_enrich_403``.
_PERMISSIVE_CAPS = {"has_members_intent": True, "has_message_content": True, "detected": False}


def _capability_disk_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "discord_capabilities.json"


def _token_cache_key(token: str) -> str:
    """Stable non-reversible cache key for a bot token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _read_caps_file(path: Path) -> Dict[str, Any]:
    """Disk cache contents ({token_key: {"caps", "ts"}}); {} when missing/corrupt."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_caps_from_disk(token: str) -> Optional[Dict[str, Any]]:
    """Return fresh disk-cached capabilities for *token*, or None."""
    try:
        entry = _read_caps_file(_capability_disk_cache_path()).get(_token_cache_key(token))
        if not isinstance(entry, dict) or time.time() - float(entry.get("ts", 0)) > _CAPABILITY_DISK_TTL_SECONDS:
            return None
        caps = entry.get("caps")
        return caps if isinstance(caps, dict) and "has_members_intent" in caps else None
    except Exception:
        return None


def _save_caps_to_disk(token: str, caps: Dict[str, Any]) -> None:
    try:
        path = _capability_disk_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _read_caps_file(path)
        data[_token_cache_key(token)] = {"caps": caps, "ts": time.time()}
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except Exception:
        logger.debug("discord capability disk-cache write failed", exc_info=True)


def _detect_capabilities_nonblocking(token: str) -> Dict[str, Any]:
    """Schema-build lookup: in-process cache → fresh disk cache → permissive default plus a
    fire-and-forget background detection that fills the disk cache for the NEXT process
    (the ~2-5s blocking HTTPS call must stay off the cold-start critical path)."""
    cached = _capability_cache.get(token)
    if cached is not None:
        return cached
    disk = _load_caps_from_disk(token)
    if disk is not None:
        _capability_cache[token] = disk
        return disk

    # Cold start — pin the permissive default for THIS process: schemas must not change
    # between agent inits within a live process or the per-conversation prompt cache breaks.
    caps_default = dict(_PERMISSIVE_CAPS)
    _capability_cache[token] = caps_default
    with _capability_bg_lock:
        if token not in _capability_bg_started:
            _capability_bg_started.add(token)

            def _bg_detect() -> None:
                try:
                    caps = _fetch_capabilities(token)
                    if caps.get("detected"):
                        _save_caps_to_disk(token, caps)
                except Exception:
                    logger.debug("background discord capability detection failed", exc_info=True)

            threading.Thread(target=_bg_detect, name="discord-caps-detect", daemon=True).start()
    return caps_default


def _fetch_capabilities(token: str) -> Dict[str, Any]:
    """Fetch capabilities from GET /applications/@me. Pure network fetch — never touches
    the in-process cache (background detection must not mutate schemas mid-process).
    Detection failure is permissive."""
    caps: Dict[str, Any] = dict(_PERMISSIVE_CAPS)
    try:
        app = _discord_request("GET", "/applications/@me", token, timeout=5)
        flags = int(app.get("flags", 0) or 0)
        caps["has_members_intent"] = bool(flags & _FLAGS_GUILD_MEMBERS)
        caps["has_message_content"] = bool(flags & _FLAGS_MESSAGE_CONTENT)
        caps["detected"] = True
    except Exception as exc:  # nosec — detection is best-effort
        logger.info("Discord capability detection failed (%s); exposing all actions.", exc)
    return caps


def _detect_capabilities(token: str, *, force: bool = False) -> Dict[str, Any]:
    """Blocking detection via GET /applications/@me, cached per token (the warm-up path;
    schema builds use the non-blocking variant). ``force`` re-fetches."""
    if token in _capability_cache and not force:
        return _capability_cache[token]
    caps = _fetch_capabilities(token)
    _capability_cache[token] = caps
    return caps


def _reset_capability_cache() -> None:
    """Test hook: clear the detection cache."""
    global _capability_cache, _capability_bg_started
    _capability_cache = {}
    with _capability_bg_lock:
        _capability_bg_started = set()


# ── action implementations ───────────────────────────────────────────────────
def _listing(key: str, items: List[Dict[str, Any]]) -> str:
    return json.dumps({key: items, "count": len(items)})


def _member_summary(m: Dict[str, Any], *, full: bool) -> Dict[str, Any]:
    """Member row; ``full`` adds the avatar/join fields member_info exposes
    (key order is part of the result text, so the two shapes stay explicit)."""
    user = m.get("user", {})
    row = {
        "user_id": user.get("id"), "username": user.get("username"), "display_name": user.get("global_name"),
        "nickname": m.get("nick"), "avatar": user.get("avatar"), "bot": user.get("bot", False),
        "roles": m.get("roles", []), "joined_at": m.get("joined_at"), "premium_since": m.get("premium_since")}
    return row if full else {k: v for k, v in row.items() if k not in ("avatar", "joined_at", "premium_since")}


def _message_summary(msg: Dict[str, Any]) -> Dict[str, Any]:
    author = msg.get("author", {})
    return {
        "id": msg["id"], "content": msg.get("content", ""),
        "author": {
            "id": author.get("id"), "username": author.get("username"),
            "display_name": author.get("global_name"), "bot": author.get("bot", False)},
        "timestamp": msg.get("timestamp"), "edited_timestamp": msg.get("edited_timestamp"),
        "attachments": [
            {"filename": a.get("filename"), "url": a.get("url"), "size": a.get("size")}
            for a in msg.get("attachments", [])],
        "reactions": [
            {"emoji": r.get("emoji", {}).get("name"), "count": r.get("count", 0)}
            for r in msg.get("reactions", [])] if msg.get("reactions") else [],
        "pinned": msg.get("pinned", False)}


def _limit_param(limit: Any, default: int) -> str:
    """Discord caps list endpoints at 100 per page."""
    try:
        return str(min(int(limit), 100))
    except (TypeError, ValueError):
        return str(min(default, 100))


def _list_guilds(token: str, **_kwargs: Any) -> str:
    guilds = _discord_request("GET", "/users/@me/guilds", token)
    return _listing("guilds", [
        {
            "id": g["id"], "name": g["name"], "icon": g.get("icon"),
            "owner": g.get("owner", False), "permissions": g.get("permissions")}
        for g in guilds])


def _server_info(token: str, guild_id: str, **_kwargs: Any) -> str:
    g = _discord_request("GET", f"/guilds/{guild_id}", token, params={"with_counts": "true"})
    return json.dumps({
        "id": g["id"], "name": g["name"], "description": g.get("description"), "icon": g.get("icon"),
        "owner_id": g.get("owner_id"), "member_count": g.get("approximate_member_count"),
        "online_count": g.get("approximate_presence_count"), "features": g.get("features", []),
        "premium_tier": g.get("premium_tier"), "premium_subscription_count": g.get("premium_subscription_count"),
        "verification_level": g.get("verification_level")})


def _list_channels(token: str, guild_id: str, **_kwargs: Any) -> str:
    """All channels grouped by category (uncategorized first), each sorted by position."""
    channels = _discord_request("GET", f"/guilds/{guild_id}/channels", token)
    cats = sorted((ch for ch in channels if ch["type"] == 4), key=lambda c: c.get("position", 0))
    groups: Dict[Optional[str], List[Dict[str, Any]]] = {None: [], **{c["id"]: [] for c in cats}}
    for ch in channels:
        if ch["type"] == 4:  # category
            continue
        parent = ch.get("parent_id")
        groups[parent if parent in groups else None].append({
            "id": ch["id"], "name": ch.get("name", ""), "type": _channel_type_name(ch["type"]),
            "position": ch.get("position", 0), "topic": ch.get("topic"), "nsfw": ch.get("nsfw", False)})
    for group in groups.values():
        group.sort(key=lambda c: c["position"])
    result = [{"category": None, "channels": groups[None]}] if groups[None] else []
    result += [{"category": {"id": c["id"], "name": c["name"]}, "channels": groups[c["id"]]} for c in cats]
    return json.dumps({"channel_groups": result, "total_channels": sum(len(g["channels"]) for g in result)})


def _channel_info(token: str, channel_id: str, **_kwargs: Any) -> str:
    ch = _discord_request("GET", f"/channels/{channel_id}", token)
    return json.dumps({
        "id": ch["id"], "name": ch.get("name"), "type": _channel_type_name(ch["type"]),
        "guild_id": ch.get("guild_id"), "topic": ch.get("topic"), "nsfw": ch.get("nsfw", False),
        "position": ch.get("position"), "parent_id": ch.get("parent_id"),
        "rate_limit_per_user": ch.get("rate_limit_per_user", 0), "last_message_id": ch.get("last_message_id")})


def _list_roles(token: str, guild_id: str, **_kwargs: Any) -> str:
    roles = _discord_request("GET", f"/guilds/{guild_id}/roles", token)
    return _listing("roles", [
        {
            "id": r["id"], "name": r["name"],
            "color": f"#{r.get('color', 0):06x}" if r.get("color") else None,
            "position": r.get("position", 0), "mentionable": r.get("mentionable", False),
            "managed": r.get("managed", False), "member_count": r.get("member_count"),
            "hoist": r.get("hoist", False)}
        for r in sorted(roles, key=lambda r: r.get("position", 0), reverse=True)])


def _member_info(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    m = _discord_request("GET", f"/guilds/{guild_id}/members/{user_id}", token)
    return json.dumps(_member_summary(m, full=True))


def _search_members(token: str, guild_id: str, query: str, limit: int = 20, **_kwargs: Any) -> str:
    """Name-prefix member search (requires the GUILD_MEMBERS intent)."""
    params = {"query": query, "limit": _limit_param(limit, 20)}
    members = _discord_request("GET", f"/guilds/{guild_id}/members/search", token, params=params)
    return _listing("members", [_member_summary(m, full=False) for m in members])


def _fetch_messages(
    token: str, channel_id: str, limit: int = 50,
    before: Optional[str] = None, after: Optional[str] = None, **_kwargs: Any) -> str:
    """``before``/``after`` are message snowflakes for reverse/forward pagination."""
    params: Dict[str, str] = {"limit": _limit_param(limit, 50)}
    if before:
        params["before"] = before
    if after:
        params["after"] = after
    messages = _discord_request("GET", f"/channels/{channel_id}/messages", token, params=params)
    return _listing("messages", [_message_summary(msg) for msg in messages])


def _list_pins(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Pinned messages (content truncated for overview)."""
    messages = _discord_request("GET", f"/channels/{channel_id}/pins", token)
    return _listing("pinned_messages", [
        {
            "id": msg["id"], "content": msg.get("content", "")[:200],
            "author": msg.get("author", {}).get("username"), "timestamp": msg.get("timestamp")}
        for msg in messages])


def _create_thread(
    token: str, channel_id: str, name: str, message_id: Optional[str] = None,
    auto_archive_duration: int = 1440, **_kwargs: Any) -> str:
    """Create a thread — anchored to ``message_id`` when given, else standalone public."""
    body: Dict[str, Any] = {"name": name, "auto_archive_duration": auto_archive_duration}
    path = f"/channels/{channel_id}/threads"
    if message_id:
        path = f"/channels/{channel_id}/messages/{message_id}/threads"
    else:
        body["type"] = 11  # PUBLIC_THREAD
    thread = _discord_request("POST", path, token, body=body)
    return json.dumps({"success": True, "thread_id": thread["id"], "name": thread.get("name")})


def _mutation(method: str, path: str, message: str):
    """Body-less write action: ``path``/``message`` are format templates over the action kwargs."""
    def _action(token: str, **kw: Any) -> str:
        _discord_request(method, path.format(**kw), token)
        return json.dumps({"success": True, "message": message.format(**kw)})
    return _action


_pin_message = _mutation("PUT", "/channels/{channel_id}/pins/{message_id}", "Message {message_id} pinned.")
_unpin_message = _mutation("DELETE", "/channels/{channel_id}/pins/{message_id}", "Message {message_id} unpinned.")
_delete_message = _mutation(
    "DELETE", "/channels/{channel_id}/messages/{message_id}", "Message {message_id} deleted.")
_add_role = _mutation(
    "PUT", "/guilds/{guild_id}/members/{user_id}/roles/{role_id}", "Role {role_id} added to user {user_id}.")
_remove_role = _mutation(
    "DELETE", "/guilds/{guild_id}/members/{user_id}/roles/{role_id}",
    "Role {role_id} removed from user {user_id}.")


def _kick_member(token: str, guild_id: str, user_id: str, reason: str = "", **_kwargs: Any) -> str:
    """Kick a member from the guild."""
    _discord_request(
        "DELETE", f"/guilds/{guild_id}/members/{user_id}", token,
        reason=reason,
    )
    return json.dumps({"success": True, "message": f"User {user_id} kicked."})


def _ban_member(
    token: str, guild_id: str, user_id: str,
    reason: str = "", delete_message_days: int = 0,
    **_kwargs: Any,
) -> str:
    """Ban a user from the guild."""
    body: Dict[str, Any] = {"delete_message_days": min(delete_message_days, 7)}
    _discord_request(
        "PUT", f"/guilds/{guild_id}/bans/{user_id}", token,
        body=body, reason=reason,
    )
    return json.dumps({"success": True, "message": f"User {user_id} banned."})


def _unban_member(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Unban a user from the guild."""
    _discord_request("DELETE", f"/guilds/{guild_id}/bans/{user_id}", token)
    return json.dumps({"success": True, "message": f"User {user_id} unbanned."})


def _timeout_member(
    token: str, guild_id: str, user_id: str,
    duration_minutes: int = 60, reason: str = "",
    **_kwargs: Any,
) -> str:
    """Timeout (mute) a guild member for a specified duration (max 28 days)."""
    from datetime import datetime, timedelta, timezone
    max_delta = timedelta(days=28)
    delta = timedelta(minutes=duration_minutes)
    if delta > max_delta:
        delta = max_delta
    until = datetime.now(timezone.utc) + delta
    body: Dict[str, Any] = {"communication_disabled_until": until.isoformat()}
    _discord_request(
        "PATCH", f"/guilds/{guild_id}/members/{user_id}", token,
        body=body, reason=reason,
    )
    return json.dumps({
        "success": True,
        "message": f"User {user_id} timed out until {until.isoformat()}.",
    })


def _remove_timeout_member(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Remove timeout from a guild member."""
    body = {"communication_disabled_until": None}
    _discord_request("PATCH", f"/guilds/{guild_id}/members/{user_id}", token, body=body)
    return json.dumps({"success": True, "message": f"Timeout removed for user {user_id}."})


def _manage_nickname(
    token: str, guild_id: str, user_id: str, nickname: str = "", **_kwargs: Any,
) -> str:
    """Change a member's nickname. Pass empty string to reset."""
    body: Dict[str, Any] = {"nick": nickname or None}
    _discord_request("PATCH", f"/guilds/{guild_id}/members/{user_id}", token, body=body)
    return json.dumps({
        "success": True,
        "message": f"Nickname for user {user_id} set to '{nickname}'.",
    })


def _change_nickname(token: str, guild_id: str, nickname: str = "", **_kwargs: Any) -> str:
    """Change the bot's own nickname in this guild."""
    body: Dict[str, Any] = {"nick": nickname or None}
    _discord_request("PATCH", f"/guilds/{guild_id}/members/@me/nick", token, body=body)
    return json.dumps({
        "success": True,
        "message": f"Bot nickname set to '{nickname}'.",
    })


def _bulk_delete_messages(
    token: str, channel_id: str, message_ids: str = "", **_kwargs: Any,
) -> str:
    """Bulk delete messages (2-100 messages). Pass message_ids as comma-separated list."""
    ids = [m.strip() for m in message_ids.split(",") if m.strip()]
    if len(ids) < 2:
        return json.dumps({
            "error": "bulk_delete_messages requires at least 2 message_ids (comma-separated).",
        })
    if len(ids) > 100:
        ids = ids[:100]
    _discord_request("POST", f"/channels/{channel_id}/messages/bulk-delete", token, body={"messages": ids})
    return json.dumps({"success": True, "message": f"Deleted {len(ids)} messages."})


# ---------------------------------------------------------------------------
# Channel management
# ---------------------------------------------------------------------------

def _create_channel(
    token: str, guild_id: str, name: str, channel_type: str = "text",
    topic: str = "", parent_id: str = "", nsfw: bool = False,
    **_kwargs: Any,
) -> str:
    """Create a new channel in a guild.

    channel_type: text (0), voice (2), announcement (5), forum (15), media (16).
    """
    type_map = {"text": 0, "voice": 2, "announcement": 5, "forum": 15, "media": 16}
    body: Dict[str, Any] = {
        "name": name,
        "type": type_map.get(channel_type, 0),
    }
    if topic:
        body["topic"] = topic
    if parent_id:
        body["parent_id"] = parent_id
    if nsfw:
        body["nsfw"] = True
    ch = _discord_request("POST", f"/guilds/{guild_id}/channels", token, body=body)
    return json.dumps({
        "success": True,
        "channel_id": ch["id"],
        "name": ch.get("name"),
        "type": _channel_type_name(ch.get("type", 0)),
    })


def _edit_channel(
    token: str, channel_id: str, name: str = "", topic: str = "",
    nsfw: Optional[bool] = None, parent_id: str = "",
    rate_limit_per_user: int = -1, **_kwargs: Any,
) -> str:
    """Edit a channel's settings. Only provided fields are changed."""
    body: Dict[str, Any] = {}
    if name:
        body["name"] = name
    if topic:
        body["topic"] = topic
    if nsfw is not None:
        body["nsfw"] = nsfw
    if parent_id:
        body["parent_id"] = parent_id
    if rate_limit_per_user >= 0:
        body["rate_limit_per_user"] = rate_limit_per_user
    if not body:
        return json.dumps({"error": "No fields to edit. Provide at least one of: name, topic, nsfw, parent_id, rate_limit_per_user."})
    ch = _discord_request("PATCH", f"/channels/{channel_id}", token, body=body)
    return json.dumps({
        "success": True,
        "channel_id": ch["id"],
        "name": ch.get("name"),
        "type": _channel_type_name(ch.get("type", 0)),
    })


def _delete_channel(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Delete a channel or category."""
    ch = _discord_request("DELETE", f"/channels/{channel_id}", token)
    return json.dumps({
        "success": True,
        "channel_id": ch.get("id") if ch else channel_id,
        "message": f"Channel {channel_id} deleted.",
    })


def _create_category(token: str, guild_id: str, name: str, **_kwargs: Any) -> str:
    """Create a new category (channel type 4) in a guild."""
    body = {"name": name, "type": 4}
    ch = _discord_request("POST", f"/guilds/{guild_id}/channels", token, body=body)
    return json.dumps({
        "success": True,
        "category_id": ch["id"],
        "name": ch.get("name"),
    })


# ---------------------------------------------------------------------------
# Role management
# ---------------------------------------------------------------------------

def _create_role(
    token: str, guild_id: str, name: str, color: str = "",
    hoist: bool = False, mentionable: bool = False,
    permissions: str = "", **_kwargs: Any,
) -> str:
    """Create a new role in a guild.

    color: hex string like '#FF0000'. permissions: optional bitfield string.
    """
    body: Dict[str, Any] = {"name": name, "hoist": hoist, "mentionable": mentionable}
    if color:
        try:
            body["color"] = int(color.lstrip("#"), 16)
        except ValueError:
            pass
    if permissions:
        body["permissions"] = permissions
    r = _discord_request("POST", f"/guilds/{guild_id}/roles", token, body=body)
    return json.dumps({
        "success": True,
        "role_id": r["id"],
        "name": r.get("name"),
        "color": f"#{r.get('color', 0):06x}" if r.get("color") else None,
    })


def _edit_role(
    token: str, guild_id: str, role_id: str, name: str = "",
    color: str = "", hoist: Optional[bool] = None,
    mentionable: Optional[bool] = None, permissions: str = "",
    **_kwargs: Any,
) -> str:
    """Edit an existing role. Only provided fields are changed."""
    body: Dict[str, Any] = {}
    if name:
        body["name"] = name
    if color:
        try:
            body["color"] = int(color.lstrip("#"), 16)
        except ValueError:
            pass
    if hoist is not None:
        body["hoist"] = hoist
    if mentionable is not None:
        body["mentionable"] = mentionable
    if permissions:
        body["permissions"] = permissions
    if not body:
        return json.dumps({"error": "No fields to edit. Provide at least one of: name, color, hoist, mentionable, permissions."})
    r = _discord_request("PATCH", f"/guilds/{guild_id}/roles/{role_id}", token, body=body)
    return json.dumps({
        "success": True,
        "role_id": r["id"],
        "name": r.get("name"),
    })


def _delete_role(token: str, guild_id: str, role_id: str, **_kwargs: Any) -> str:
    """Delete a role from a guild."""
    _discord_request("DELETE", f"/guilds/{guild_id}/roles/{role_id}", token)
    return json.dumps({"success": True, "message": f"Role {role_id} deleted."})


# ---------------------------------------------------------------------------
# Webhook management
# ---------------------------------------------------------------------------

def _list_webhooks(token: str, channel_id: str = "", guild_id: str = "", **_kwargs: Any) -> str:
    """List webhooks in a channel or guild. Provide channel_id OR guild_id."""
    if channel_id:
        hooks = _discord_request("GET", f"/channels/{channel_id}/webhooks", token)
    elif guild_id:
        hooks = _discord_request("GET", f"/guilds/{guild_id}/webhooks", token)
    else:
        return json.dumps({"error": "Provide either channel_id or guild_id."})
    result = []
    for h in hooks:
        result.append({
            "id": h["id"],
            "name": h.get("name"),
            "channel_id": h.get("channel_id"),
            "avatar": h.get("avatar"),
            "token": bool(h.get("token")),
        })
    return json.dumps({"webhooks": result, "count": len(result)})


def _create_webhook(token: str, channel_id: str, name: str, **_kwargs: Any) -> str:
    """Create a new webhook in a channel."""
    body = {"name": name}
    h = _discord_request("POST", f"/channels/{channel_id}/webhooks", token, body=body)
    return json.dumps({
        "success": True,
        "webhook_id": h["id"],
        "name": h.get("name"),
        "token": h.get("token"),
        "url": f"https://discord.com/api/v10/webhooks/{h['id']}/{h.get('token', '')}" if h.get("token") else None,
    })


def _delete_webhook(token: str, webhook_id: str, **_kwargs: Any) -> str:
    """Delete a webhook."""
    _discord_request("DELETE", f"/webhooks/{webhook_id}", token)
    return json.dumps({"success": True, "message": f"Webhook {webhook_id} deleted."})


# ---------------------------------------------------------------------------
# Message editing & reactions
# ---------------------------------------------------------------------------

def _edit_message(
    token: str, channel_id: str, message_id: str,
    content: str = "", embed_json: str = "",
    **_kwargs: Any,
) -> str:
    """Edit an existing message in a channel or thread."""
    body: Dict[str, Any] = {}
    if content:
        body["content"] = content
    if embed_json:
        body["embeds"] = json.loads(embed_json)
    if not body:
        return json.dumps({"error": "No edit fields provided (content or embed_json required)."})
    msg = _discord_request("PATCH", f"/channels/{channel_id}/messages/{message_id}", token, body=body)
    return json.dumps({"success": True, "message_id": msg["id"], "content": msg.get("content", "")})


def _add_reaction(token: str, channel_id: str, message_id: str, emoji: str, **_kwargs: Any) -> str:
    """Add a reaction (emoji) to a message. Emoji can be Unicode or :name:id format."""
    # URL-encode emoji for the path
    encoded = urllib.parse.quote(emoji, safe="")
    _discord_request("PUT", f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me", token)
    return json.dumps({"success": True, "message": f"Reaction {emoji} added to message {message_id}."})


def _remove_reaction(token: str, channel_id: str, message_id: str, emoji: str, **_kwargs: Any) -> str:
    """Remove the bot's own reaction from a message."""
    encoded = urllib.parse.quote(emoji, safe="")
    _discord_request("DELETE", f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me", token)
    return json.dumps({"success": True, "message": f"Reaction {emoji} removed from message {message_id}."})


# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------

def _archive_thread(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Archive a thread (channel_id is the thread ID)."""
    _discord_request("PATCH", f"/channels/{channel_id}", token, body={"archived": True})
    return json.dumps({"success": True, "message": f"Thread {channel_id} archived."})


def _unarchive_thread(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Unarchive a thread (channel_id is the thread ID)."""
    _discord_request("PATCH", f"/channels/{channel_id}", token, body={"archived": False})
    return json.dumps({"success": True, "message": f"Thread {channel_id} unarchived."})


def _delete_thread(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Delete a thread permanently (channel_id is the thread ID)."""
    _discord_request("DELETE", f"/channels/{channel_id}", token)
    return json.dumps({"success": True, "message": f"Thread {channel_id} deleted."})


def _list_thread_members(token: str, channel_id: str, **_kwargs: Any) -> str:
    """List members of a thread (channel_id is the thread ID)."""
    members = _discord_request("GET", f"/channels/{channel_id}/thread-members", token)
    result = []
    for m in members:
        result.append({
            "user_id": m["user_id"],
            "join_timestamp": m.get("join_timestamp"),
            "flags": m.get("flags"),
        })
    return json.dumps({"members": result, "count": len(result)})


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _get_audit_log(
    token: str, guild_id: str, limit: int = 50,
    action_type: int = 0, user_id: str = "",
    **_kwargs: Any,
) -> str:
    """Fetch the guild's audit log.

    action_type: filter by event type (see Discord docs). 0 means no filter.
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    params: Dict[str, str] = {"limit": str(min(limit, 100))}
    if action_type:
        params["action_type"] = str(action_type)
    if user_id:
        params["user_id"] = user_id
    log = _discord_request("GET", f"/guilds/{guild_id}/audit-logs", token, params=params)
    entries = log.get("audit_log_entries", [])
    result = []
    for entry in entries:
        result.append({
            "id": entry["id"],
            "action_type": entry.get("action_type"),
            "user_id": entry.get("user_id"),
            "target_id": entry.get("target_id"),
            "reason": entry.get("reason"),
            "created_at": entry.get("created_at"),
        })
    return json.dumps({
        "entries": result,
        "count": len(result),
        "users": [{"id": u["id"], "username": u.get("username")} for u in log.get("users", [])],
    })


# ---------------------------------------------------------------------------
# Scheduled events
# ---------------------------------------------------------------------------

def _list_scheduled_events(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List scheduled events in a guild."""
    events = _discord_request("GET", f"/guilds/{guild_id}/scheduled-events", token, params={"with_user_count": "true"})
    result = []
    for ev in events:
        result.append({
            "id": ev["id"],
            "name": ev.get("name"),
            "description": ev.get("description"),
            "scheduled_start_time": ev.get("scheduled_start_time"),
            "scheduled_end_time": ev.get("scheduled_end_time"),
            "privacy_level": ev.get("privacy_level"),
            "status": ev.get("status"),
            "entity_type": ev.get("entity_type"),
            "entity_id": ev.get("entity_id"),
            "creator_id": ev.get("creator_id"),
            "user_count": ev.get("user_count"),
        })
    return json.dumps({"events": result, "count": len(result)})


def _create_scheduled_event(
    token: str, guild_id: str, name: str,
    scheduled_start_time: str = "",
    scheduled_end_time: str = "",
    event_type: str = "voice",
    channel_id: str = "",
    description: str = "",
    location: str = "",
    **_kwargs: Any,
) -> str:
    """Create a scheduled event in a guild.

    event_type: "stage" (1), "voice" (2), or "external" (3).
    channel_id required for stage/voice; location required for external.
    """
    type_map = {"stage": 1, "voice": 2, "external": 3}
    entity_type = type_map.get(event_type.lower(), 2)

    body: Dict[str, Any] = {
        "name": name,
        "privacy_level": 2,
        "scheduled_start_time": scheduled_start_time,
        "entity_type": entity_type,
    }
    if scheduled_end_time:
        body["scheduled_end_time"] = scheduled_end_time
    if description:
        body["description"] = description

    if entity_type == 3:
        if not location:
            return json.dumps({"error": "External events require a location."})
        if not scheduled_end_time:
            return json.dumps({"error": "External events require a scheduled_end_time."})
        body["entity_metadata"] = {"location": location}
        body["scheduled_end_time"] = scheduled_end_time
    else:
        if not channel_id:
            return json.dumps({"error": "Stage/Voice events require a channel_id."})
        body["channel_id"] = channel_id

    ev = _discord_request("POST", f"/guilds/{guild_id}/scheduled-events", token, body=body)
    return json.dumps({
        "success": True,
        "event_id": ev["id"],
        "name": ev.get("name"),
        "scheduled_start_time": ev.get("scheduled_start_time"),
    })


def _edit_scheduled_event(
    token: str, guild_id: str, event_id: str,
    name: str = "",
    scheduled_start_time: str = "",
    scheduled_end_time: str = "",
    description: str = "",
    channel_id: str = "",
    location: str = "",
    **_kwargs: Any,
) -> str:
    """Edit an existing scheduled event. Only provided fields are changed."""
    body: Dict[str, Any] = {}
    if name:
        body["name"] = name
    if scheduled_start_time:
        body["scheduled_start_time"] = scheduled_start_time
    if scheduled_end_time:
        body["scheduled_end_time"] = scheduled_end_time
    if description:
        body["description"] = description
    if channel_id:
        body["channel_id"] = channel_id
    if location:
        body["entity_metadata"] = {"location": location}

    if not body:
        return json.dumps({"error": "No edit fields provided."})

    ev = _discord_request("PATCH", f"/guilds/{guild_id}/scheduled-events/{event_id}", token, body=body)
    return json.dumps({
        "success": True,
        "event_id": ev["id"],
        "name": ev.get("name"),
    })


def _delete_scheduled_event(token: str, guild_id: str, event_id: str, **_kwargs: Any) -> str:
    """Delete a scheduled event."""
    _discord_request("DELETE", f"/guilds/{guild_id}/scheduled-events/{event_id}", token)
    return json.dumps({"success": True, "message": f"Scheduled event {event_id} deleted."})




# ---------------------------------------------------------------------------
# Messaging
# ---------------------------------------------------------------------------

def _send_message(
    token: str, channel_id: str, content: str,
    reply_to_message_id: str = "",
    **_kwargs: Any,
) -> str:
    """Send a message to a channel or thread."""
    if not content:
        return json.dumps({"error": "content is required to send a message."})
    body: Dict[str, Any] = {"content": content}
    if reply_to_message_id:
        body["message_reference"] = {"message_id": reply_to_message_id}
    msg = _discord_request("POST", f"/channels/{channel_id}/messages", token, body=body)
    return json.dumps({
        "success": True,
        "message_id": msg["id"],
        "channel_id": msg.get("channel_id", channel_id),
    })


def _crosspost_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Crosspost a message from an announcement channel to subscribed channels."""
    msg = _discord_request("POST", f"/channels/{channel_id}/messages/{message_id}/crosspost", token)
    return json.dumps({
        "success": True,
        "message_id": msg.get("id", message_id),
        "message": f"Message {message_id} crossposted.",
    })


def _create_poll(
    token: str, channel_id: str,
    poll_question: str = "",
    poll_answers: str = "",
    poll_duration: int = 24,
    poll_multiselect: bool = False,
    **_kwargs: Any,
) -> str:
    """Send a poll in a channel. poll_answers is a comma-separated list (2-10 items)."""
    if not poll_question:
        return json.dumps({"error": "poll_question is required."})
    answers = [a.strip() for a in poll_answers.split(",") if a.strip()]
    if len(answers) < 2 or len(answers) > 10:
        return json.dumps({"error": f"Poll requires 2-10 answers, got {len(answers)}."})
    body: Dict[str, Any] = {
        "poll": {
            "question": {"text": poll_question},
            "answers": [{"poll_media": {"text": a}} for a in answers],
            "duration": poll_duration,
            "allow_multiselect": poll_multiselect,
        }
    }
    msg = _discord_request("POST", f"/channels/{channel_id}/messages", token, body=body)
    return json.dumps({
        "success": True,
        "message_id": msg["id"],
        "poll": True,
    })



# ---------------------------------------------------------------------------
# Voice state management
# ---------------------------------------------------------------------------

def _mute_member(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Mute a member in voice channels (server-mute)."""
    _discord_request("PATCH", f"/guilds/{guild_id}/members/{user_id}", token, body={"mute": True})
    return json.dumps({"success": True, "message": f"User {user_id} muted."})


def _unmute_member(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Unmute a member in voice channels."""
    _discord_request("PATCH", f"/guilds/{guild_id}/members/{user_id}", token, body={"mute": False})
    return json.dumps({"success": True, "message": f"User {user_id} unmuted."})


def _move_member(token: str, guild_id: str, user_id: str, channel_id: str, **_kwargs: Any) -> str:
    """Move a member to a different voice channel."""
    if not channel_id:
        return json.dumps({"error": "channel_id is required to move a member."})
    _discord_request("PATCH", f"/guilds/{guild_id}/members/{user_id}", token, body={"channel_id": channel_id})
    return json.dumps({"success": True, "message": f"User {user_id} moved to channel {channel_id}."})


def _disconnect_member(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Disconnect a member from their current voice channel."""
    _discord_request("PATCH", f"/guilds/{guild_id}/members/{user_id}", token, body={"channel_id": None})
    return json.dumps({"success": True, "message": f"User {user_id} disconnected."})

# ---------------------------------------------------------------------------
# Action dispatch + metadata
# ---------------------------------------------------------------------------


# ── action dispatch + metadata ───────────────────────────────────────────────
# Single source of truth: (action, handler, required-param signature, description). Order is
# the schema/enum order; the signature drives runtime required-param validation.
_ACTION_MANIFEST = [
    ("list_guilds", _list_guilds, "()", "list servers the bot is in"),
    ("server_info", _server_info, "(guild_id)", "server details + member counts"),
    ("list_channels", _list_channels, "(guild_id)", "all channels grouped by category"),
    ("channel_info", _channel_info, "(channel_id)", "single channel details"),
    ("list_roles", _list_roles, "(guild_id)", "roles sorted by position"),
    ("member_info", _member_info, "(guild_id, user_id)", "lookup a specific member"),
    ("search_members", _search_members, "(guild_id, query)", "find members by name prefix"),
    ("fetch_messages", _fetch_messages, "(channel_id)", "recent messages; optional before/after snowflakes"),
    ("list_pins", _list_pins, "(channel_id)", "pinned messages in a channel"),
    ("pin_message", _pin_message, "(channel_id, message_id)", "pin a message"),
    ("unpin_message", _unpin_message, "(channel_id, message_id)", "unpin a message"),
    ("delete_message", _delete_message, "(channel_id, message_id)", "delete a message"),
    ("create_thread", _create_thread, "(channel_id, name)", "create a public thread; optional message_id anchor"),
    ("add_role", _add_role, "(guild_id, user_id, role_id)", "assign a role"),
    ("remove_role", _remove_role, "(guild_id, user_id, role_id)", "remove a role"),
    ("bulk_delete_messages", _bulk_delete_messages, "(channel_id, message_ids)", "bulk delete 2-100 messages (comma-separated IDs)"),
    ("kick_member", _kick_member, "(guild_id, user_id)", "remove a member from the server"),
    ("ban_member", _ban_member, "(guild_id, user_id)", "ban a user; optional delete_message_days (0-7)"),
    ("unban_member", _unban_member, "(guild_id, user_id)", "unban a previously banned user"),
    ("timeout_member", _timeout_member, "(guild_id, user_id)", "timeout/mute a member; optional duration_minutes (max 40320)"),
    ("remove_timeout_member", _remove_timeout_member, "(guild_id, user_id)", "remove timeout from a member"),
    ("change_nickname", _change_nickname, "(guild_id, nickname)", "change the bot's own nickname"),
    ("manage_nickname", _manage_nickname, "(guild_id, user_id, nickname)", "change a member's nickname"),
    ("create_channel", _create_channel, "(guild_id, name)", "create a new channel; optional channel_type/topic/parent_id/nsfw"),
    ("edit_channel", _edit_channel, "(channel_id)", "edit channel settings; optional name/topic/nsfw/parent_id/rate_limit_per_user"),
    ("delete_channel", _delete_channel, "(channel_id)", "delete a channel or category"),
    ("create_category", _create_category, "(guild_id, name)", "create a new category"),
    ("create_role", _create_role, "(guild_id, name)", "create a role; optional color/hoist/mentionable/permissions"),
    ("edit_role", _edit_role, "(guild_id, role_id)", "edit a role; optional name/color/hoist/mentionable/permissions"),
    ("delete_role", _delete_role, "(guild_id, role_id)", "delete a role"),
    ("list_webhooks", _list_webhooks, "()", "list webhooks in a channel or guild; provide channel_id OR guild_id"),
    ("create_webhook", _create_webhook, "(channel_id, name)", "create a new webhook in a channel"),
    ("delete_webhook", _delete_webhook, "(webhook_id)", "delete a webhook"),
    ("get_audit_log", _get_audit_log, "(guild_id)", "fetch audit log; optional limit/action_type/user_id"),
    ("edit_message", _edit_message, "(channel_id, message_id)", "edit a message; optional content/embed_json"),
    ("add_reaction", _add_reaction, "(channel_id, message_id, emoji)", "add an emoji reaction to a message"),
    ("remove_reaction", _remove_reaction, "(channel_id, message_id, emoji)", "remove a bot emoji reaction from a message"),
    ("archive_thread", _archive_thread, "(channel_id)", "archive a thread"),
    ("unarchive_thread", _unarchive_thread, "(channel_id)", "unarchive a thread"),
    ("delete_thread", _delete_thread, "(channel_id)", "delete a thread"),
    ("list_thread_members", _list_thread_members, "(channel_id)", "list members of a thread"),
    ("list_scheduled_events", _list_scheduled_events, "(guild_id)", "list scheduled events in a guild"),
    ("create_scheduled_event", _create_scheduled_event, "(guild_id, name, scheduled_start_time)", "create a scheduled event; optional event_type/channel_id/location/description"),
    ("edit_scheduled_event", _edit_scheduled_event, "(guild_id, event_id)", "edit an event; optional name/scheduled_start_time/scheduled_end_time/location/description"),
    ("delete_scheduled_event", _delete_scheduled_event, "(guild_id, event_id)", "delete a scheduled event"),
    ("send_message", _send_message, "(channel_id, content)", "send a message to a channel; optional reply_to_message_id"),
    ("crosspost_message", _crosspost_message, "(channel_id, message_id)", "crosspost a message from an announcement channel"),
    ("create_poll", _create_poll, "(channel_id, poll_question, poll_answers)", "send a poll; optional poll_duration/poll_multiselect"),
    ("mute_member", _mute_member, "(guild_id, user_id)", "server-mute a member in voice channels"),
    ("unmute_member", _unmute_member, "(guild_id, user_id)", "remove server-mute from a member"),
    ("move_member", _move_member, "(guild_id, user_id, channel_id)", "move a member to a different voice channel"),
    ("disconnect_member", _disconnect_member, "(guild_id, user_id)", "disconnect a member from voice channels"),
]
_ACTIONS = {name: fn for name, fn, _sig, _desc in _ACTION_MANIFEST}
_REQUIRED_PARAMS: Dict[str, List[str]] = {
    name: [p.strip() for p in sig.strip("()").split(",") if p.strip()]
    for name, _fn, sig, _desc in _ACTION_MANIFEST}

# Two tools share one action table: ``discord`` (core, the participation trio every bot
# user wants) and ``discord_admin`` (everything else).
_CORE_ACTION_NAMES = frozenset({"fetch_messages", "search_members", "create_thread"})
_CORE_ACTIONS = {k: v for k, v in _ACTIONS.items() if k in _CORE_ACTION_NAMES}
_ADMIN_ACTIONS = {k: v for k, v in _ACTIONS.items() if k not in _CORE_ACTION_NAMES}

# Actions that require the GUILD_MEMBERS privileged intent.
_INTENT_GATED_MEMBERS = frozenset({"member_info", "search_members"})


def _load_allowed_actions_config() -> Optional[List[str]]:
    """``discord.server_actions`` allowlist (comma string or YAML list), or ``None`` when
    unrestricted. Unknown names are dropped with a warning."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as exc:
        logger.debug("discord: could not load config (%s); allowing all actions.", exc)
        return None
    raw = (cfg.get("discord") or {}).get("server_actions")
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        raw = raw.split(",")
    elif not isinstance(raw, (list, tuple)):
        logger.warning("discord.server_actions: unexpected type %s; ignoring.", type(raw).__name__)
        return None
    names = [str(n).strip() for n in raw if str(n).strip()]
    invalid = [n for n in names if n not in _ACTIONS]
    if invalid:
        logger.warning(
            "discord.server_actions: unknown action(s) ignored: %s. Known: %s",
            ", ".join(invalid), ", ".join(_ACTIONS.keys()))
    return [n for n in names if n in _ACTIONS]


def _available_actions(caps: Dict[str, Any], allowlist: Optional[List[str]]) -> List[str]:
    """Visible actions from intents + config allowlist, in :data:`_ACTIONS` order."""
    members_ok = caps.get("has_members_intent", True)
    return [
        name for name in _ACTIONS
        if (members_ok or name not in _INTENT_GATED_MEMBERS) and (allowlist is None or name in allowlist)]


# ── schema construction ──────────────────────────────────────────────────────
_TOOL_DESCRIPTIONS = {
    "discord_admin": (
        "Manage a Discord server via the REST API.",
        "Call list_guilds first to discover guild_ids, then list_channels for "
        "channel_ids. Runtime errors will tell you if the bot lacks a specific "
        "per-guild permission (e.g. MANAGE_ROLES for add_role).",
    ),
    "discord": (
        "Read and participate in a Discord server.",
        "Use the channel_id from the current conversation context. "
        "Use search_members to look up user IDs by name prefix.",
    ),
}

_SCHEMA_PROPERTIES: Dict[str, Any] = {
    "guild_id": {"type": "string", "description": "Discord server (guild) ID."},
    "channel_id": {"type": "string", "description": "Discord channel ID."},
    "user_id": {"type": "string", "description": "Discord user ID."},
    "role_id": {"type": "string", "description": "Discord role ID."},
    "message_id": {"type": "string", "description": "Discord message ID."},
    "query": {"type": "string", "description": "Member name prefix to search for (search_members)."},
    "name": {"type": "string", "description": "New thread name (create_thread)."},
    "limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "description": "Max results (default 50). Applies to fetch_messages, search_members.",
    },
    "before": {"type": "string", "description": "Snowflake ID for reverse pagination (fetch_messages)."},
    "after": {"type": "string", "description": "Snowflake ID for forward pagination (fetch_messages)."},
    "auto_archive_duration": {
        "type": "integer",
        "enum": [60, 1440, 4320, 10080],
        "description": "Thread archive duration in minutes (create_thread, default 1440).",
    },
    "content": {
        "type": "string",
        "description": "Message content (send_message, edit_message).",
    },
    "embed_json": {
        "type": "string",
        "description": "JSON string of embed objects (edit_message).",
    },
    "reply_to_message_id": {
        "type": "string",
        "description": "Message ID to reply to (send_message).",
    },
    "reason": {
        "type": "string",
        "description": "Audit-log reason shown in server audit log (kick_member, ban_member, timeout_member).",
    },
    "delete_message_days": {
        "type": "integer",
        "minimum": 0,
        "maximum": 7,
        "description": "Days of messages to delete when banning (ban_member, default 0).",
    },
    "duration_minutes": {
        "type": "integer",
        "minimum": 1,
        "maximum": 40320,
        "description": "Timeout duration in minutes, max 40320 (28 days) (timeout_member, default 60).",
    },
    "nickname": {
        "type": "string",
        "description": "New nickname; empty string resets it (change_nickname, manage_nickname).",
    },
    "message_ids": {
        "type": "string",
        "description": "Comma-separated message IDs for bulk_delete_messages.",
    },
    "channel_type": {
        "type": "string",
        "enum": ["text", "voice", "announcement", "forum", "media"],
        "description": "Channel type for create_channel (default 'text').",
    },
    "topic": {
        "type": "string",
        "description": "Channel topic (create_channel, edit_channel).",
    },
    "nsfw": {
        "type": "boolean",
        "description": "Mark channel as NSFW (create_channel, edit_channel).",
    },
    "parent_id": {
        "type": "string",
        "description": "Parent category ID (create_channel, edit_channel).",
    },
    "rate_limit_per_user": {
        "type": "integer",
        "minimum": 0,
        "maximum": 21600,
        "description": "Slowmode duration in seconds (edit_channel). -1 means no change.",
    },
    "color": {
        "type": "string",
        "description": "Role color hex string like '#FF0000' (create_role, edit_role).",
    },
    "hoist": {
        "type": "boolean",
        "description": "Display role members separately in sidebar (create_role, edit_role).",
    },
    "mentionable": {
        "type": "boolean",
        "description": "Allow @mentioning this role (create_role, edit_role).",
    },
    "permissions": {
        "type": "string",
        "description": "Permission bitfield string (create_role, edit_role).",
    },
    "webhook_id": {
        "type": "string",
        "description": "Webhook ID (delete_webhook).",
    },
    "action_type": {
        "type": "integer",
        "description": "Audit log event type filter, 0 = no filter (get_audit_log).",
    },
    "emoji": {
        "type": "string",
        "description": "Emoji for reaction (Unicode like '👍' or custom ':name:id').",
    },
    "event_id": {
        "type": "string",
        "description": "Scheduled event ID (edit_scheduled_event, delete_scheduled_event).",
    },
    "event_type": {
        "type": "string",
        "enum": ["stage", "voice", "external"],
        "description": "Event type for create_scheduled_event (default 'voice').",
    },
    "location": {
        "type": "string",
        "description": "Physical location for external events (create_scheduled_event).",
    },
    "scheduled_start_time": {
        "type": "string",
        "description": "ISO 8601 start time, e.g. 2024-12-31T20:00:00Z (create_scheduled_event).",
    },
    "scheduled_end_time": {
        "type": "string",
        "description": "ISO 8601 end time, optional (create_scheduled_event).",
    },
    "poll_question": {
        "type": "string",
        "description": "Poll question text, max 300 chars (create_poll).",
    },
    "poll_answers": {
        "type": "string",
        "description": "Comma-separated poll answers, 2-10 items, max 55 chars each (create_poll).",
    },
    "poll_duration": {
        "type": "integer",
        "minimum": 1,
        "maximum": 168,
        "description": "Poll duration in hours, max 168 (create_poll, default 24).",
    },
    "poll_multiselect": {
        "type": "boolean",
        "description": "Allow multiple selections (create_poll, default false).",
    },

}

_CONTENT_NOTE = (
    "\n\nNOTE: Bot does NOT have the MESSAGE_CONTENT privileged intent. "
    "{names} will return message metadata (author, "
    "timestamps, attachments, reactions, pin state) but `content` will be "
    "empty for messages not sent as a direct mention to the bot or in DMs. "
    "Enable the intent in the Discord Developer Portal to see all content."
)


def _build_schema(
    actions: List[str], caps: Optional[Dict[str, Any]] = None, tool_name: str = "discord",
) -> Optional[Dict[str, Any]]:
    """Tool schema for the filtered action list; ``None`` when empty (drop the tool)."""
    caps = caps or {}
    if not actions:
        return None
    manifest_block = "\n".join(
        f"  {name}{sig}  — {desc}" for name, _fn, sig, desc in _ACTION_MANIFEST if name in actions)
    content_note = ""
    affected_actions = {"fetch_messages", "list_pins"} & set(actions)
    if affected_actions and caps.get("detected") and caps.get("has_message_content") is False:
        content_note = _CONTENT_NOTE.format(names=" and ".join(sorted(affected_actions)))
    lead, guidance = _TOOL_DESCRIPTIONS.get(tool_name, _TOOL_DESCRIPTIONS["discord"])
    return {
        "name": tool_name,
        "description": f"{lead}\n\nAvailable actions:\n{manifest_block}\n\n{guidance}{content_note}",
        "parameters": {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": actions}, **_SCHEMA_PROPERTIES},
            "required": ["action"]}}


def _get_dynamic_schema(action_subset: Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
    """Build a dynamic schema for *action_subset* filtered by intents + config."""
    token = _get_bot_token()
    if not token:
        return None
    caps = _detect_capabilities_nonblocking(token)
    actions = [a for a in _available_actions(caps, _load_allowed_actions_config()) if a in action_subset]
    return _build_schema(actions, caps, tool_name=tool_name) if actions else None


get_dynamic_schema_core = functools.partial(_get_dynamic_schema, _CORE_ACTIONS, "discord")
get_dynamic_schema_admin = functools.partial(_get_dynamic_schema, _ADMIN_ACTIONS, "discord_admin")


# ── 403 error enrichment ─────────────────────────────────────────────────────
_NO_MANAGE_MESSAGES = "Bot lacks MANAGE_MESSAGES permission in this channel"
_VIEW_HISTORY = "Bot cannot view this channel (missing VIEW_CHANNEL or READ_MESSAGE_HISTORY)."
_ROLE_HIERARCHY = "Either the bot lacks MANAGE_ROLES, or the target role sits higher than the bot's highest role."

# Per-action guidance for a call-time 403 (per-guild permissions are never pre-checked).
_ACTION_403_HINT = {
    "pin_message": (
        f"{_NO_MANAGE_MESSAGES}. "
        "Ask the server admin to grant the bot a role that has MANAGE_MESSAGES, or a per-channel overwrite."),
    "unpin_message": f"{_NO_MANAGE_MESSAGES}.",
    "delete_message": f"{_NO_MANAGE_MESSAGES}, or cannot view the channel/message.",
    "create_thread": "Bot lacks CREATE_PUBLIC_THREADS in this channel, or cannot view it.",
    "add_role": (
        f"{_ROLE_HIERARCHY} Roles can only be assigned below the bot's own position in the role hierarchy."),
    "remove_role": _ROLE_HIERARCHY,
    "fetch_messages": _VIEW_HISTORY,
    "list_pins": _VIEW_HISTORY,
    "channel_info": "Bot cannot view this channel (missing VIEW_CHANNEL).",
    "search_members": (
        "Likely missing the Server Members privileged intent — enable it in the Discord Developer Portal "
        "under your bot's settings."),
    "member_info": "Bot cannot see this guild member (missing Server Members intent or insufficient permissions).",
    "bulk_delete_messages": (
        "Bot lacks MANAGE_MESSAGES permission in this channel."
    ),
    "kick_member": (
        "Bot lacks KICK_MEMBERS permission, or the target is the server owner / "
        "has a role higher than the bot."
    ),
    "ban_member": (
        "Bot lacks BAN_MEMBERS permission, or the target is the server owner / "
        "has a role higher than the bot."
    ),
    "unban_member": (
        "Bot lacks BAN_MEMBERS permission."
    ),
    "timeout_member": (
        "Bot lacks MODERATE_MEMBERS permission, or the target has a role higher "
        "than the bot."
    ),
    "remove_timeout_member": (
        "Bot lacks MODERATE_MEMBERS permission."
    ),
    "change_nickname": (
        "Bot lacks CHANGE_NICKNAME permission (rare — usually means the server "
        "has disabled nickname changes)."
    ),
    "manage_nickname": (
        "Bot lacks MANAGE_NICKNAMES permission, or the target has a role higher "
        "than the bot."
    ),
    "create_channel": (
        "Bot lacks MANAGE_CHANNELS permission in this guild."
    ),
    "edit_channel": (
        "Bot lacks MANAGE_CHANNELS permission, or cannot view this channel."
    ),
    "delete_channel": (
        "Bot lacks MANAGE_CHANNELS permission, or cannot view this channel."
    ),
    "create_category": (
        "Bot lacks MANAGE_CHANNELS permission in this guild."
    ),
    "create_role": (
        "Bot lacks MANAGE_ROLES permission in this guild, or its highest role is not above the new role."
    ),
    "edit_role": (
        "Bot lacks MANAGE_ROLES permission, or the target role sits higher than the bot's highest role."
    ),
    "delete_role": (
        "Bot lacks MANAGE_ROLES permission, or the target role sits higher than the bot's highest role."
    ),
    "list_webhooks": (
        "Bot lacks MANAGE_WEBHOOKS permission, or cannot view the target channel/guild."
    ),
    "create_webhook": (
        "Bot lacks MANAGE_WEBHOOKS permission in this channel."
    ),
    "delete_webhook": (
        "Bot lacks MANAGE_WEBHOOKS permission, or the webhook belongs to a channel the bot cannot manage."
    ),
    "get_audit_log": (
        "Bot lacks VIEW_AUDIT_LOG permission in this guild."
    ),
    "edit_message": (
        "Bot lacks MANAGE_MESSAGES permission, or the message is too old (>24h)."
    ),
    "add_reaction": (
        "Bot lacks ADD_REACTIONS permission, or the emoji is invalid/unavailable."
    ),
    "remove_reaction": (
        "Bot lacks ADD_REACTIONS permission, or the reaction does not exist."
    ),
    "archive_thread": (
        "Bot lacks MANAGE_THREADS permission in this thread."
    ),
    "unarchive_thread": (
        "Bot lacks MANAGE_THREADS permission in this thread."
    ),
    "delete_thread": (
        "Bot lacks MANAGE_THREADS permission in this thread."
    ),
    "list_thread_members": (
        "Bot cannot view this thread, or it has not joined the thread."
    ),
    "list_scheduled_events": (
        "Bot lacks VIEW_GUILD_INSIGHTS or cannot access this guild."
    ),
    "create_scheduled_event": (
        "Bot lacks CREATE_EVENTS permission in this guild."
    ),
    "edit_scheduled_event": (
        "Bot lacks MANAGE_EVENTS permission, or the event no longer exists."
    ),
    "delete_scheduled_event": (
        "Bot lacks MANAGE_EVENTS permission, or the event no longer exists."
    ),
    "send_message": (
        "Bot lacks SEND_MESSAGES permission in this channel, or has been rate-limited."
    ),
    "crosspost_message": (
        "Bot lacks SEND_MESSAGES, or the channel is not an announcement channel, or the message was already crossposted."
    ),
    "create_poll": (
        "Bot lacks SEND_MESSAGES or SEND_POLLS permission in this channel."
    ),
    "mute_member": (
        "Bot lacks MUTE_MEMBERS permission, or the target user has a higher role."
    ),
    "unmute_member": (
        "Bot lacks MUTE_MEMBERS permission, or the target user has a higher role."
    ),
    "move_member": (
        "Bot lacks MOVE_MEMBERS permission, or the target voice channel is inaccessible."
    ),
    "disconnect_member": (
        "Bot lacks MOVE_MEMBERS permission, or the user is not in a voice channel."
    ),
}


def _enrich_403(action: str, body: str) -> str:
    """Return a user-friendly guidance string for a 403 on ``action``."""
    hint = _ACTION_403_HINT.get(action)
    base = f"Discord API 403 (forbidden) on '{action}'."
    return f"{base} {hint} (Raw: {body})" if hint else f"{base} (Raw: {body})"


def check_discord_tool_requirements() -> bool:
    """Tool is available only when a Discord bot token is configured."""
    return bool(_get_bot_token())


# ── handlers ─────────────────────────────────────────────────────────────────
_HANDLER_DEFAULTS = {
    "guild_id": "", "channel_id": "", "user_id": "", "role_id": "", "message_id": "", "query": "",
    "name": "", "limit": 50, "before": "", "after": "", "auto_archive_duration": 1440,
    "content": "", "embed_json": "", "reply_to_message_id": "", "reason": "",
    "delete_message_days": 0, "duration_minutes": 60, "nickname": "", "message_ids": "",
    "channel_type": "text", "topic": "", "nsfw": False, "parent_id": "",
    "rate_limit_per_user": -1, "color": "", "hoist": False, "mentionable": False,
    "permissions": "", "webhook_id": "", "action_type": 0, "emoji": "",
    "event_id": "", "event_type": 2, "location": "", "scheduled_start_time": "",
    "scheduled_end_time": "", "poll_question": "", "poll_answers": "",
    "poll_duration": 24, "poll_multiselect": False}


def _run_discord_action(action: str, valid_actions: Dict[str, Any], tool_label: str, **params: Any) -> str:
    """Shared handler logic for both discord tools (``params`` default per :data:`_HANDLER_DEFAULTS`)."""
    token = _get_bot_token()
    if not token:
        return tool_error("DISCORD_BOT_TOKEN not configured.")
    action_fn = valid_actions.get(action)
    if not action_fn:
        return tool_error(f"Unknown action: {action}", available_actions=list(valid_actions.keys()))
    # Config-level allowlist gate (defense in depth): a stale cached schema from a prior
    # config must not let denied actions through.
    allowlist = _load_allowed_actions_config()
    if allowlist is not None and action not in allowlist:
        return tool_error(
            f"Action '{action}' is disabled by config (discord.server_actions). "
            f"Allowed: {', '.join(allowlist) if allowlist else '<none>'}")
    kwargs = {k: params.get(k, v) for k, v in _HANDLER_DEFAULTS.items()}
    missing = [p for p in _REQUIRED_PARAMS.get(action, []) if not kwargs.get(p)]
    if missing:
        return tool_error(f"Missing required parameters for '{action}': {', '.join(missing)}")
    try:
        return action_fn(token=token, **kwargs)
    except DiscordAPIError as e:
        logger.warning("Discord API error in %s action '%s': %s", tool_label, action, e)
        return tool_error(_enrich_403(action, e.body) if e.status == 403 else str(e))
    except Exception as e:
        logger.exception("Unexpected error in %s action '%s'", tool_label, action)
        return tool_error(f"Unexpected error: {e}")


# ``discord`` = core participation trio; ``discord_admin`` = server management.
discord_core = functools.partial(_run_discord_action, valid_actions=_CORE_ACTIONS, tool_label="discord")
discord_admin_handler = functools.partial(
    _run_discord_action, valid_actions=_ADMIN_ACTIONS, tool_label="discord_admin")


# Static (un-detected) schemas at import; the intent/config-filtered ones come from
# get_dynamic_schema_core/admin via model_tools' dynamic schema overrides.
for _name, _actions, _handler in (
    ("discord", _CORE_ACTIONS, discord_core), ("discord_admin", _ADMIN_ACTIONS, discord_admin_handler),
):
    registry.register(
        name=_name,
        toolset=_name,
        schema=_build_schema(list(_actions), caps={"detected": False}, tool_name=_name),
        handler=lambda args, _h=_handler, **kw: _h(**{"action": "", **args}),
        check_fn=check_discord_tool_requirements,
        requires_env=["DISCORD_BOT_TOKEN"])


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import TYPE_CHECKING  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402

def get_dynamic_schema() -> Optional[Dict[str, Any]]:
    """Backward-compat wrapper — returns core schema."""
    return get_dynamic_schema_core()
# ---- END PLUGIN-COMPAT ----

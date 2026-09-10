from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from app.history.store import ChatHistoryStore
from app.llm.artifacts.store import ArtifactUriError, LocalArtifactStore


ArtifactReadMode = Literal["summary", "text", "json", "search"]

MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_TEXT_LINES = 200
MAX_JSON_ITEMS = 20
MAX_SEARCH_MATCHES = 10
MAX_CONTEXT_LINES = 3
MAX_OUTPUT_CHARACTERS = 30_000
MAX_MATCH_CHARACTERS = 2_500


class ArtifactAccessError(ValueError):
    """An artifact is invalid, unavailable, or outside the caller's scope."""


class ArtifactAccessService:
    """Authorize and bound reads of locally stored JSON artifacts."""

    def __init__(
        self,
        history_store: ChatHistoryStore,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.history_store = history_store
        self.artifact_store = artifact_store or LocalArtifactStore()

    async def read(
        self,
        *,
        artifact_uri: str,
        user_id: str,
        mode: ArtifactReadMode,
        start_line: int = 1,
        max_lines: int = 100,
        json_pointer: str | None = None,
        offset: int = 0,
        limit: int = 20,
        query: str | None = None,
        context_lines: int = 0,
        max_matches: int = 10,
    ) -> dict[str, Any]:
        try:
            run_id, path = self.artifact_store.resolve_json_uri(artifact_uri)
        except ArtifactUriError as exc:
            raise ArtifactAccessError("artifact is unavailable") from exc

        scope = await self.history_store.get_run_scope(run_id)
        if scope is None or scope.user_id != user_id:
            raise ArtifactAccessError("artifact is unavailable")

        text = await asyncio.to_thread(self._read_text, path)
        result: dict[str, Any] = {
            "artifact_uri": artifact_uri,
            "run_id": run_id,
            "mode": mode,
        }
        if mode == "text":
            result["data"] = self._read_lines(text, start_line, max_lines)
        elif mode == "search":
            if not query:
                raise ArtifactAccessError("search query is required")
            result["data"] = self._search_lines(
                text,
                query,
                context_lines,
                max_matches,
            )
        else:
            payload = self._parse_json(text)
            result["data"] = (
                self._summarize(payload)
                if mode == "summary"
                else self._select_json(payload, json_pointer, offset, limit)
            )
        return result

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            if path.stat().st_size > MAX_ARTIFACT_BYTES:
                raise ArtifactAccessError("artifact is too large")
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ArtifactAccessError("artifact is unavailable") from exc

    @staticmethod
    def _parse_json(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ArtifactAccessError("artifact is not valid JSON") from exc

    @staticmethod
    def _clip(value: str, maximum: int) -> tuple[str, bool]:
        if len(value) <= maximum:
            return value, False
        return f"{value[:maximum]}\n…", True

    def _read_lines(
        self,
        text: str,
        start_line: int,
        max_lines: int,
    ) -> dict[str, Any]:
        lines = text.splitlines()
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        content, clipped = self._clip("\n".join(selected), MAX_OUTPUT_CHARACTERS)
        return {
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1 if selected else None,
            "total_lines": len(lines),
            "content": content,
            "truncated": clipped or start_line - 1 + len(selected) < len(lines),
        }

    def _search_lines(
        self,
        text: str,
        query: str,
        context_lines: int,
        max_matches: int,
    ) -> dict[str, Any]:
        lines = text.splitlines()
        matches: list[dict[str, Any]] = []
        needle = query.casefold()
        for index, line in enumerate(lines):
            if needle not in line.casefold():
                continue
            start = max(0, index - context_lines)
            end = min(len(lines), index + context_lines + 1)
            context, clipped = self._clip(
                "\n".join(lines[start:end]),
                MAX_MATCH_CHARACTERS,
            )
            matches.append(
                {
                    "line": index + 1,
                    "context_start_line": start + 1,
                    "context": context,
                    "truncated": clipped,
                }
            )
            if len(matches) == max_matches:
                break
        return {
            "query": query,
            "matches": matches,
            "truncated": len(matches) == max_matches,
        }

    def _select_json(
        self,
        payload: Any,
        json_pointer: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        selected = self._resolve_pointer(payload, json_pointer or "")
        result: dict[str, Any] = {"json_pointer": json_pointer or ""}
        if isinstance(selected, list):
            result.update(
                {
                    "total_items": len(selected),
                    "offset": offset,
                    "items": selected[offset : offset + limit],
                    "truncated": offset + limit < len(selected),
                }
            )
        else:
            result["value"] = selected

        encoded = json.dumps(result, ensure_ascii=False, default=str)
        if len(encoded) <= MAX_OUTPUT_CHARACTERS:
            return result
        preview, _ = self._clip(encoded, MAX_OUTPUT_CHARACTERS)
        return {"json_preview": preview, "truncated": True}

    @staticmethod
    def _resolve_pointer(payload: Any, pointer: str) -> Any:
        if not pointer:
            return payload
        if not pointer.startswith("/") or re.search(r"~(?![01])", pointer):
            raise ArtifactAccessError("invalid JSON Pointer")

        current = payload
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(current, Mapping):
                if token not in current:
                    raise ArtifactAccessError("JSON Pointer does not exist")
                current = current[token]
            elif isinstance(current, list) and token.isdigit() and (
                token == "0" or not token.startswith("0")
            ):
                index = int(token)
                if index >= len(current):
                    raise ArtifactAccessError("JSON Pointer does not exist")
                current = current[index]
            else:
                raise ArtifactAccessError("JSON Pointer does not exist")
        return current

    @staticmethod
    def _summarize(payload: Any) -> dict[str, Any]:
        if isinstance(payload, Mapping):
            item_counts = {
                str(key): len(value)
                for key, value in payload.items()
                if isinstance(value, (dict, list))
            }
            return {
                "type": "object",
                "keys": list(payload.keys()),
                "item_counts": item_counts,
            }
        if isinstance(payload, list):
            return {"type": "array", "total_items": len(payload)}
        return {"type": type(payload).__name__, "value": payload}

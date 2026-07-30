from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.config import settings
from app.infrastructure.lightrag_client import LightRAGClient
from app.infrastructure.paper_rag_index_grpc_client import (
    PaperRagIndexGrpcClient,
    paper_rag_index_grpc_client,
)
from app.llm.artifacts.store import LocalArtifactStore


class RagIndexRunner:
    """Upload queued paper PDFs to LightRAG and persist a result artifact."""

    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        lightrag_client: LightRAGClient | None = None,
        rag_index_client: PaperRagIndexGrpcClient | None = None,
        poll_interval_seconds: float | None = None,
        max_polls: int | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.lightrag_client = lightrag_client or LightRAGClient()
        self.rag_index_client = rag_index_client or paper_rag_index_grpc_client
        self.poll_interval_seconds = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else settings.LIGHTRAG_INDEX_POLL_INTERVAL_SECONDS
        )
        self.max_polls = max_polls if max_polls is not None else settings.LIGHTRAG_INDEX_MAX_POLLS
        self.lease_seconds = settings.LIGHTRAG_INDEX_LEASE_SECONDS
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"
        self._tasks: set[asyncio.Task[None]] = set()

    def enqueue(self, request_artifact_uri: str) -> None:
        task = asyncio.create_task(self._run_background(request_artifact_uri))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run(self, request_artifact_uri: str) -> str:
        manifest = await self._read_artifact(request_artifact_uri)
        run_id = manifest.get("run_id")
        requests = manifest.get("requests")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("RAG index request manifest is missing run_id")
        if not isinstance(requests, list):
            raise ValueError("RAG index request manifest requests must be a list")

        results = [await self._index_request(item) for item in requests]
        artifact = await self.artifact_store.write_json(
            run_id=run_id,
            step_key="run_rag_index",
            source="lightrag",
            kind="rag_index_result_manifest_json",
            payload={
                "run_id": run_id,
                "search_task_id": manifest.get("search_task_id"),
                "input_manifest": request_artifact_uri,
                "results": results,
            },
            count=sum(item["status"] == "ready" for item in results),
        )
        return artifact.artifact_uri

    async def close(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait(self) -> None:
        """Wait for all tasks submitted before this call to finish."""
        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.gather(*tasks)

    async def _run_background(self, request_artifact_uri: str) -> None:
        try:
            await self.run(request_artifact_uri)
        except Exception:
            return

    async def _index_request(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            return {"status": "failed", "error": "invalid RAG index request"}

        paper_id = request.get("paper_id")
        source_pdf_path = request.get("source_pdf_path")
        result: dict[str, Any] = {"paper_id": paper_id, "status": "failed"}
        if not isinstance(source_pdf_path, str) or not source_pdf_path.strip():
            skipped = await self.rag_index_client.skip(
                [paper_id],
                reason="missing local PDF path",
            )
            result.update(
                status="skipped" if skipped["ok"] else "failed",
                error=None if skipped["ok"] else skipped["error"],
            )
            return result

        try:
            content_hash = await asyncio.to_thread(self._content_hash, source_pdf_path)
            item = {"paper_id": paper_id, "content_hash": content_hash}
            ensured = await self.rag_index_client.ensure([item])
            if not ensured["ok"]:
                result["error"] = ensured["error"]
                return result
            ensure_result = ensured["result"]["indexes"][0]
            existing = ensure_result["index"]
            if not ensure_result["should_index"]:
                result.update(existing, reused=True, status=existing["index_status"])
                return result

            claimed = await self.rag_index_client.claim(
                [item],
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
            )
            if not claimed["ok"]:
                result["error"] = claimed["error"]
                return result
            claims = claimed["result"]["claimed"]
            if not claims:
                result.update(status="indexing", error="paper index is claimed by another worker")
                return result
            lease_token = claims[0]["lease_token"]

            track_id = await self.lightrag_client.upload_file(source_pdf_path)
            status, track_payload = await self._wait_for_terminal_status(track_id)
        except Exception as exc:
            if "lease_token" in locals() and "content_hash" in locals():
                await self.rag_index_client.complete(
                    paper_id=paper_id,
                    lease_token=lease_token,
                    content_hash=content_hash,
                    index_status="failed",
                    error_message=str(exc),
                )
            result["error"] = str(exc)
            return result

        result.update(track_id=track_id, track_status=status)
        document_id = str(
            track_payload.get("document_id")
            or track_payload.get("doc_id")
            or ""
        )
        if status == "processed":
            result["status"] = "ready"
            completed = await self.rag_index_client.complete(
                paper_id=paper_id,
                lease_token=lease_token,
                content_hash=content_hash,
                index_status="ready",
                lightrag_track_id=track_id,
                lightrag_document_id=document_id,
            )
        elif status == "failed":
            completed = await self.rag_index_client.complete(
                paper_id=paper_id,
                lease_token=lease_token,
                content_hash=content_hash,
                index_status="failed",
                lightrag_track_id=track_id,
                error_message="LightRAG document processing failed",
            )
            result["error"] = "LightRAG document processing failed"
        else:
            completed = await self.rag_index_client.complete(
                paper_id=paper_id,
                lease_token=lease_token,
                content_hash=content_hash,
                index_status="failed",
                lightrag_track_id=track_id,
                error_message="LightRAG processing timed out",
            )
            result.update(status="failed", error="LightRAG processing timed out")
        if not completed["ok"] or not completed["result"]["accepted"]:
            result["error"] = completed["error"] or "paper RAG index completion was not accepted"
        if document_id:
            result["lightrag_document_id"] = document_id
        return result

    async def _wait_for_terminal_status(self, track_id: str) -> tuple[str, dict[str, Any]]:
        for attempt in range(self.max_polls):
            payload = await self.lightrag_client.get_track_status(track_id)
            status = str(payload.get("status") or "").strip().casefold()
            if status in {"processed", "failed"}:
                return status, payload
            if attempt + 1 < self.max_polls:
                await asyncio.sleep(self.poll_interval_seconds)
        return "timeout", {}

    @staticmethod
    def _content_hash(source_pdf_path: str) -> str:
        digest = hashlib.sha256()
        with Path(source_pdf_path).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    async def _read_artifact(self, artifact_uri: str) -> dict[str, Any]:
        if not artifact_uri.startswith("artifact://"):
            raise ValueError("RAG index request artifact must use artifact:// URI")
        base_dir = self.artifact_store.base_dir.resolve()
        path = (base_dir / artifact_uri.removeprefix("artifact://")).resolve()
        path.relative_to(base_dir)

        def read() -> dict[str, Any]:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                raise ValueError("RAG index request manifest must be a JSON object")
            return payload

        return await asyncio.to_thread(read)


rag_index_runner = RagIndexRunner()

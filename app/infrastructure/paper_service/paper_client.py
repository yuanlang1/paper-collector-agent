import json
from datetime import date, datetime, time, timezone
from typing import Any

from app.infrastructure.paper_service.http_client import (
    CLIENT_ERRORS,
    ServiceClient,
    list_value,
    object_value,
    positive_id,
    text_value,
)


class PaperServiceClient(ServiceClient):
    @staticmethod
    def _list_text(value: Any) -> str:
        if isinstance(value, list):
            return json.dumps(
                [str(item).strip() for item in value if str(item).strip()],
                ensure_ascii=False,
            )
        return str(value or "").strip()

    @staticmethod
    def _timestamp(value: Any) -> str | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            resolved = value
        elif isinstance(value, date):
            resolved = datetime.combine(value, time.min)
        elif isinstance(value, str):
            resolved = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            raise ValueError("published_date must be an ISO-8601 date or datetime")
        if resolved.tzinfo is None:
            resolved = resolved.replace(tzinfo=timezone.utc)
        return resolved.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @classmethod
    def _build_request(cls, item: dict[str, Any]) -> dict[str, Any]:
        client_key = str(item["client_key"]).strip()
        paper = object_value(item["paper_info"])
        title = str(paper.get("title") or "").strip()
        authors = cls._list_text(paper.get("authors"))
        if not client_key:
            raise ValueError("client_key must not be blank")
        if not title or not authors or authors == "[]":
            raise ValueError("title and authors must not be blank")
        venue_id = paper.get("venue_id")
        if type(venue_id) is not int or venue_id < 0:
            raise ValueError("venue_id must be a non-negative integer")
        citations = paper.get("citations")
        if isinstance(citations, bool):
            raise ValueError("citations must be an integer")
        return {
            "clientKey": client_key,
            "title": title,
            "authors": authors,
            "paperAbstract": str(
                paper.get("paper_abstract") or paper.get("abstract") or ""
            ),
            "aiAbstract": str(paper.get("ai_abstract") or ""),
            "doi": str(paper.get("doi") or ""),
            "venueId": venue_id,
            "keywords": cls._list_text(paper.get("keywords")),
            "source": str(paper.get("source") or ""),
            "pdfUrl": str(paper.get("pdf_url") or ""),
            "abstractUrl": str(paper.get("abstract_url") or ""),
            "ossName": str(paper.get("oss_name") or ""),
            "publishedDate": cls._timestamp(paper.get("published_date")),
            "citations": int(citations) if citations is not None else None,
        }

    async def batch_save_papers(self, papers: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            if not papers:
                raise ValueError("papers must not be empty")
            requests = [self._build_request(paper) for paper in papers]
            response = await self.http.request(
                "POST", "/internal/paper/batch-save", json={"papers": requests},
            )
            saved = [object_value(item) for item in list_value(response["data"])]
            if len(saved) != len(requests) or any(
                item.get("clientKey") != request["clientKey"]
                for item, request in zip(saved, requests)
            ):
                raise ValueError("Saved papers do not match request client keys")
            return self._success(
                {
                    "papers": [
                        {
                            "client_key": item["clientKey"],
                            "paper_id": positive_id(item["id"], "paper_id"),
                            "status": "saved",
                        }
                        for item in saved
                    ]
                },
                response,
            )
        except CLIENT_ERRORS as exc:
            return self._error(exc)

    async def get_task_review_papers(self, task_id: int) -> dict[str, Any]:
        try:
            positive_id(task_id)
            response = await self.http.request(
                "GET", f"/internal/paper/task/{task_id}/review-papers"
            )
            data = object_value(response["data"])
            papers = []
            for value in list_value(data["papers"]):
                item = object_value(value)
                papers.append(
                    {
                        "paper_id": positive_id(item["paperId"], "paper_id"),
                        "title": item.get("title") or "",
                        "authors": [
                            text_value(author).strip()
                            for author in list_value(item.get("authors") or [])
                            if text_value(author).strip()
                        ],
                        "published_date": self._timestamp(item.get("publishedDate")),
                        "doi": item.get("doi") or None,
                        "rag_status": (text_value(item.get("ragStatus")) or "PENDING")
                        .strip()
                        .lower(),
                        "chunk_count": item.get("chunkCount") or 0,
                        "paper_abstract": text_value(item.get("paperAbstract")).strip(),
                        "oss_name": text_value(item.get("ossName")).strip() or None,
                        "pdf_url": text_value(item.get("pdfUrl")).strip() or None,
                    }
                )
            return self._success(
                {
                    "task_id": data["taskId"],
                    "task_status": data["taskStatus"],
                    "papers": papers,
                },
                response,
            )
        except CLIENT_ERRORS as exc:
            return self._error(exc)


paper_service_client = PaperServiceClient()

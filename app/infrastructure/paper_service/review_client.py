from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from app.infrastructure.paper_service.http_client import (
    CLIENT_ERRORS,
    ServiceClient,
    object_value,
    positive_id,
)


REVIEW_TYPE_CODES = {
    "narrative": 1,
    "systematic": 2,
    "scoping": 3,
    "critical": 4,
}

CITATION_STYLE_CODES = {
    "harvard": 1,
    "apa": 2,
    "ieee": 3,
    "chicago": 4,
    "vancouver": 5,
}


class ReviewServiceClient(ServiceClient):
    @staticmethod
    def _required_text(payload: Mapping[str, Any], field: str,) -> str:
        value = str(payload.get(field) or "").strip()
        if not value:
            raise ValueError(f"{field} must not be blank")
        return value

    @staticmethod
    def _positive_int(value: Any, field: str,) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a positive integer")

        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a positive integer") from exc

        if parsed <= 0:
            raise ValueError(f"{field} must be a positive integer")
        return parsed

    @staticmethod
    def _json_text(value: Any, field: str) -> str:
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} must contain valid JSON") from exc
        else:
            decoded = value

        try:
            return json.dumps(decoded, ensure_ascii=False, separators=(",", ":"),)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be JSON serializable") from exc

    @classmethod
    def _paper_ids_json(cls, value: Any, field: str) -> str:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} must contain valid JSON") from exc

        if not isinstance(value, list):
            raise ValueError(f"{field} must be a JSON array")

        return cls._json_text(
            [cls._positive_int(paper_id, field) for paper_id in value], field,
        )

    @classmethod
    def _citation_labels_json(cls, value: Any) -> str:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("citation_labels must contain valid JSON") from exc

        if not isinstance(value, dict):
            raise ValueError("citation_labels must be a JSON object")

        normalized = {}
        for paper_id, label in value.items():
            normalized[str(cls._positive_int(paper_id, "citation_labels key"))] = str(
                label
            ).strip()

        if any(not label for label in normalized.values()):
            raise ValueError("citation_labels values must not be blank")
        return cls._json_text(normalized, "citation_labels")

    @classmethod
    def _references_json(cls, value: Any) -> str:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("references must contain valid JSON") from exc

        if not isinstance(value, list):
            raise ValueError("references must be a JSON array")

        normalized = []
        for reference in value:
            if not isinstance(reference, Mapping):
                raise ValueError("references items must be JSON objects")
            normalized.append(
                {
                    "paper_id": cls._positive_int(
                        reference.get("paper_id"), "references.paper_id",
                    ),
                    "formatted": cls._required_text(reference, "formatted"),
                }
            )

        return cls._json_text(normalized, "references")

    @classmethod
    def _build_add_review_request(cls, review: Mapping[str, Any],) -> dict[str, Any]:
        review_type = cls._required_text(review, "review_type").lower()
        citation_style = cls._required_text(review, "citation_style").lower()

        if review_type not in REVIEW_TYPE_CODES:
            raise ValueError(f"unsupported review_type: {review_type}")
        if citation_style not in CITATION_STYLE_CODES:
            raise ValueError(f"unsupported citation_style: {citation_style}")

        return {
            "taskId": cls._positive_int(review.get("task_id"), "task_id"),
            "topic": cls._required_text(review, "topic"),
            "title": cls._required_text(review, "title"),
            "versionNumber": cls._positive_int(
                review.get("version_number", 1), "version_number",
            ),
            "language": cls._required_text(review, "language"),
            "reviewType": REVIEW_TYPE_CODES[review_type],
            "citationStyle": CITATION_STYLE_CODES[citation_style],
            "scope": str(review.get("scope") or "").strip(),
            "abstractContent": cls._required_text(review, "abstract_content"),
            "bodyMarkdown": cls._required_text(review, "body_markdown"),
            "conclusion": cls._required_text(review, "conclusion"),
            "markdown": cls._required_text(review, "markdown"),
            "sections": cls._json_text(review.get("sections"), "sections"),
            "paperIdsSnapshot": cls._paper_ids_json(
                review.get("paper_ids_snapshot"), "paper_ids_snapshot",
            ),
            "citationPaperIds": cls._paper_ids_json(
                review.get("citation_paper_ids"), "citation_paper_ids",
            ),
            "citationLabels": cls._citation_labels_json(review.get("citation_labels"),),
            "references": cls._references_json(review.get("references")),
        }

    async def add_review(self, review: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = await self.http.request(
                "POST",
                "/internal/review/add",
                json=self._build_add_review_request(review),
            )
            data = object_value(response["data"])
            return self._success(
                {
                    "review_id": positive_id(data["reviewId"], "review_id"),
                    "version_number": positive_id(
                        data["versionNumber"], "version_number"
                    ),
                },
                response,
            )
        except CLIENT_ERRORS as exc:
            return self._error(exc)


review_service_client = ReviewServiceClient()

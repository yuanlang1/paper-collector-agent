from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.citations import (
    citation_paper_ids as cited_paper_ids,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed


class FinalizingHandoffNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        reflection_report, review_draft, corpus = await asyncio.gather(
            self.artifact_store.read_json_uri(
                state["reflection_report_artifact_ref"]
            ),
            self.artifact_store.read_json_uri(
                state["review_draft_artifact_ref"]
            ),
            self.artifact_store.read_json_uri(
                state["corpus_artifact_ref"]
            ),
        )

        if (
            reflection_report["effective_decision"] != "pass"
            or reflection_report["review_draft_artifact_ref"]
            != state["review_draft_artifact_ref"]
        ):
            return failed("review was not approved by reflection")

        final_review = self._build_final_review(
            review_draft=review_draft,
            corpus=corpus,
            citation_style=state["citation_style"],
            reflection_report_artifact_ref=(
                state["reflection_report_artifact_ref"]
            ),
        )

        artifact = await self.artifact_store.write_json(
            run_id=state["child_run_id"],
            step_key="finalizing_handoff",
            source="task_review",
            kind="task_review_final_json",
            count=len(final_review["citation_paper_ids"]),
            payload=final_review,
        )

        return {
            "final_review_artifact_ref": artifact.artifact_uri,
            "stage": "persisting_review",
            "status": "running",
            "error": None,
        }

    def _build_final_review(
        self,
        *,
        review_draft: dict[str, Any],
        corpus: dict[str, Any],
        citation_style: str,
        reflection_report_artifact_ref: str,
    ) -> dict[str, Any]:
        raw_text = "\n".join(
            [
                review_draft["abstract"],
                review_draft["body_markdown"],
                review_draft["conclusion"],
            ]
        )

        citation_paper_ids = cited_paper_ids(raw_text)

        papers_by_id = {
            str(paper["paper_id"]): paper
            for paper in corpus["papers"]
        }

        citation_labels = {
            paper_id: f"paper-{paper_id}"
            for paper_id in citation_paper_ids
        }

        references = [
            {
                "paper_id": paper_id,
                "formatted": self._reference(
                    paper=papers_by_id[paper_id],
                    number=index,
                    citation_style=citation_style,
                ),
                "title": papers_by_id[paper_id]["title"],
                "authors": papers_by_id[paper_id]["authors"],
                "published_date": papers_by_id[paper_id][
                    "published_date"
                ],
                "doi": papers_by_id[paper_id]["doi"],
            }
            for index, paper_id in enumerate(
                citation_paper_ids,
                start=1,
            )
        ]

        markdown = "\n\n".join(
            part
            for part in [
                f"# {review_draft['title']}",
                review_draft["abstract"],
                review_draft["body_markdown"],
                review_draft["conclusion"],
            ]
            if part
        )

        return {
            "task_id": review_draft["task_id"],
            "title": review_draft["title"],
            "scope": review_draft["scope"],
            "abstract": review_draft["abstract"],
            "sections": review_draft["sections"],
            "body_markdown": review_draft["body_markdown"],
            "conclusion": review_draft["conclusion"],
            "markdown": markdown,
            "citation_style": citation_style,
            "citation_paper_ids": citation_paper_ids,
            "citation_labels": citation_labels,
            "references": references,
            "reflection_report_artifact_ref": (
                reflection_report_artifact_ref
            ),
        }

    def _reference(
        self,
        *,
        paper: dict[str, Any],
        number: int,
        citation_style: str,
    ) -> str:
        authors = ", ".join(paper["authors"])
        title = paper["title"]
        year = self._year(paper["published_date"])
        doi = (
            f" https://doi.org/{paper['doi']}"
            if paper["doi"]
            else ""
        )

        if citation_style == "ieee":
            author_prefix = f"{authors}, " if authors else ""
            return f'[{number}] {author_prefix}"{title}," {year}.{doi}'

        if citation_style == "vancouver":
            author_prefix = f"{authors}. " if authors else ""
            return f"{number}. {author_prefix}{title}. {year}.{doi}"

        if citation_style == "apa":
            if authors:
                return f"{authors} ({year}). {title}.{doi}"
            return f"{title}. ({year}).{doi}"

        if citation_style == "chicago":
            if authors:
                return f"{authors}. {year}. {title}.{doi}"
            return f"{title}. {year}.{doi}"

        if authors:
            return f"{authors} ({year}) {title}.{doi}"
        return f"{title} ({year}).{doi}"

    def _year(
        self,
        published_date: str | None,
    ) -> str:
        match = re.search(
            r"\b(18|19|20)\d{2}\b",
            published_date or "",
        )
        return match.group(0) if match else "n.d."

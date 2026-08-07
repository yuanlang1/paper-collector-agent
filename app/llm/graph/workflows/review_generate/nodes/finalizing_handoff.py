from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore


REF_PATTERN = re.compile(r"\[\[REF_(\d+)\]\]")


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

        if not reflection_report["satisfied"]:
            return {
                "stage": "failed",
                "status": "failed",
                "error": "review was not approved by reflection",
            }

        final_review = self._build_final_review(
            review_draft=review_draft,
            corpus=corpus,
            citation_style=state["citation_style"],
            language=state["language"],
            reflection_report_artifact_ref=(
                state["reflection_report_artifact_ref"]
            ),
        )

        artifact = await self.artifact_store.write_json(
            run_id=state["run_id"],
            step_key="finalizing_handoff",
            source="task_review",
            kind="task_review_final_json",
            count=len(final_review["citation_paper_ids"]),
            payload=final_review,
        )

        return {
            "final_review_artifact_ref": artifact.artifact_uri,
            "handoff": {
                "action": "upsert_review",
                "task_id": state["task_id"],
                "review_artifact_ref": artifact.artifact_uri,
                "title": final_review["title"],
                "status": "completed",
                "citation_paper_ids": (
                    final_review["citation_paper_ids"]
                ),
            },
            "stage": "completed",
            "status": "completed",
            "error": None,
        }

    def _build_final_review(
        self,
        *,
        review_draft: dict[str, Any],
        corpus: dict[str, Any],
        citation_style: str,
        language: str,
        reflection_report_artifact_ref: str,
    ) -> dict[str, Any]:
        raw_text = "\n".join(
            [
                review_draft["abstract"],
                review_draft["body_markdown"],
                review_draft["conclusion"],
            ]
        )

        citation_paper_ids = list(
            dict.fromkeys(
                REF_PATTERN.findall(raw_text)
            )
        )

        papers_by_id = {
            str(paper["paper_id"]): paper
            for paper in corpus["papers"]
        }

        citation_labels = {
            paper_id: self._in_text_citation(
                paper=papers_by_id[paper_id],
                number=index,
                citation_style=citation_style,
            )
            for index, paper_id in enumerate(
                citation_paper_ids,
                start=1,
            )
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

        abstract = self._replace_anchors(
            review_draft["abstract"],
            citation_labels,
        )
        body_markdown = self._replace_anchors(
            review_draft["body_markdown"],
            citation_labels,
        )
        conclusion = self._replace_anchors(
            review_draft["conclusion"],
            citation_labels,
        )

        sections = [
            {
                **section,
                "text": self._replace_anchors(
                    section["text"],
                    citation_labels,
                ),
            }
            for section in review_draft["sections"]
        ]

        headings = self._headings(language)
        references_markdown = "\n".join(
            reference["formatted"]
            for reference in references
        )

        markdown = "\n\n".join(
            [
                f"# {review_draft['title']}",
                f"## {headings['abstract']}\n\n{abstract}",
                body_markdown,
                f"## {headings['conclusion']}\n\n{conclusion}",
                f"## {headings['references']}\n\n{references_markdown}",
            ]
        )

        return {
            "task_id": review_draft["task_id"],
            "title": review_draft["title"],
            "scope": review_draft["scope"],
            "abstract": abstract,
            "sections": sections,
            "body_markdown": body_markdown,
            "conclusion": conclusion,
            "markdown": markdown,
            "citation_style": citation_style,
            "citation_paper_ids": citation_paper_ids,
            "citation_labels": citation_labels,
            "references": references,
            "reflection_report_artifact_ref": (
                reflection_report_artifact_ref
            ),
        }

    def _replace_anchors(
        self,
        text: str,
        citation_labels: dict[str, str],
    ) -> str:
        return REF_PATTERN.sub(
            lambda match: citation_labels[match.group(1)],
            text,
        )

    def _in_text_citation(
        self,
        *,
        paper: dict[str, Any],
        number: int,
        citation_style: str,
    ) -> str:
        if citation_style == "ieee":
            return f"[{number}]"

        if citation_style == "vancouver":
            return f"({number})"

        author = self._author_key(
            paper["authors"],
            paper["title"],
            citation_style,
        )
        return f"({author}, {self._year(paper['published_date'])})"

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

    def _author_key(
        self,
        authors: list[str],
        title: str,
        citation_style: str,
    ) -> str:
        surnames = [
            author.split()[-1]
            for author in authors
        ]

        if not surnames:
            return title

        if len(surnames) == 1:
            return surnames[0]

        if len(surnames) == 2:
            separator = " & " if citation_style == "apa" else " and "
            return separator.join(surnames)

        return f"{surnames[0]} et al."

    def _year(
        self,
        published_date: str | None,
    ) -> str:
        match = re.search(
            r"\b(18|19|20)\d{2}\b",
            published_date or "",
        )
        return match.group(0) if match else "n.d."

    def _headings(
        self,
        language: str,
    ) -> dict[str, str]:
        if language == "zh-CN":
            return {
                "abstract": "摘要",
                "conclusion": "结论",
                "references": "参考文献",
            }

        return {
            "abstract": "Abstract",
            "conclusion": "Conclusion",
            "references": "References",
        }

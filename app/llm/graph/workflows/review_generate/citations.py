from __future__ import annotations

import re


CITATION_KEY_PATTERN = re.compile(r"@paper-(\d+)")


def citation_paper_ids(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_KEY_PATTERN.findall(text)))


def unsupported_synthesis_citation_ids(
    *,
    body_markdown: str,
    abstract: str,
    conclusion: str,
) -> set[str]:
    return set(citation_paper_ids(abstract) + citation_paper_ids(conclusion)) - set(
        citation_paper_ids(body_markdown)
    )

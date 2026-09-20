from __future__ import annotations

import re


REF_PATTERN = re.compile(r"\[\[REF_([^\]]+)\]\]")


def citation_anchor_ids(text: str) -> list[str]:
    return list(dict.fromkeys(REF_PATTERN.findall(text)))


def unsupported_synthesis_anchor_ids(
    *,
    body_markdown: str,
    abstract: str,
    conclusion: str,
) -> set[str]:
    return set(citation_anchor_ids(abstract) + citation_anchor_ids(conclusion)) - set(
        citation_anchor_ids(body_markdown)
    )

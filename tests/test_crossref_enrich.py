import unittest

from app.llm.graph.workflows.paper_search.nodes.crossref_enrich import (
    CrossrefMetadataEnrichmentNode,
)


class CrossrefAbstractEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.node = CrossrefMetadataEnrichmentNode()

    def test_fills_a_missing_original_abstract(self):
        paper = {"paper_info": {"paper_abstract": None}}

        self.node._merge(
            paper,
            {"abstract": "Abstract provided by Crossref."},
            "doi",
        )

        self.assertEqual(
            paper["paper_info"]["paper_abstract"],
            "Abstract provided by Crossref.",
        )

    def test_does_not_overwrite_an_existing_original_abstract(self):
        paper = {"paper_info": {"paper_abstract": "Original source abstract."}}

        self.node._merge(
            paper,
            {"abstract": "Abstract provided by Crossref."},
            "doi",
        )

        self.assertEqual(
            paper["paper_info"]["paper_abstract"],
            "Original source abstract.",
        )


if __name__ == "__main__":
    unittest.main()

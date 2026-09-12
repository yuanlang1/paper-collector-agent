import json
import unittest

from langchain_core.documents import Document

from app.infrastructure.mineru.mineru_client import MinerUClient
from app.rag.data_preparation.data_preparation import DataPreparationModule
from app.rag.file_parser.file_parser import FileParser
from app.rag.index_construction.paper_content_index_construction import (
    PaperContentIndexConstructionModule,
)
from app.rag.retrieval.paper_content_retrieval import (
    PaperContentHybridRetrievalModule,
)
from app.schemas.file_parser import PdfSourceBlock


class PdfPageAwareChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = Document(
            page_content = "parent markdown",
            metadata = {
                "paper_id": "7",
                "parent_id": "7",
            },
        )

    def test_content_list_pages_are_one_based_and_survive_retrieval(self) -> None:
        content_list = MinerUClient.extract_content_list(
            {
                "result/paper_content_list.json": json.dumps(
                    [
                        {
                            "type": "text",
                            "text": "alpha",
                            "page_idx": 0,
                            "bbox": [1, 2, 3, 4],
                        },
                        {
                            "type": "text",
                            "text": "bravo",
                            "page_idx": 1,
                        },
                    ]
                ).encode(),
            }
        )
        blocks = FileParser._build_pdf_source_blocks(content_list)
        chunks = DataPreparationModule(
            chunk_size = 12,
            chunk_overlap = 0,
        ).prepare_pdf_document(self.document, blocks)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["page_numbers"], [1, 2])
        self.assertEqual(chunks[0].metadata["page_start"], 1)
        self.assertEqual(chunks[0].metadata["page_end"], 2)
        self.assertEqual(
            chunks[0].metadata["source_spans"][0]["bbox"],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            chunks[0].metadata["source_spans"],
            [
                {
                    "page": 1,
                    "type": "text",
                    "source_index": 0,
                    "char_start": 0,
                    "char_end": 5,
                    "bbox": [1, 2, 3, 4],
                },
                {
                    "page": 2,
                    "type": "text",
                    "source_index": 1,
                    "char_start": 7,
                    "char_end": 12,
                },
            ],
        )

        index = object.__new__(PaperContentIndexConstructionModule)
        payload = index._build_payload(chunks[0])
        retrieval = object.__new__(PaperContentHybridRetrievalModule)
        retrieved = retrieval._payload_to_document(payload)

        self.assertEqual(retrieved.metadata["page_numbers"], [1, 2])
        self.assertEqual(retrieved.metadata["page_start"], 1)
        self.assertEqual(
            retrieved.metadata["source_spans"][1]["char_start"],
            7,
        )

    def test_long_text_block_keeps_its_source_page(self) -> None:
        chunks = DataPreparationModule(
            chunk_size = 12,
            chunk_overlap = 3,
        ).prepare_pdf_document(
            self.document,
            [
                PdfSourceBlock(
                    markdown = "one two three four five",
                    page_no = 3,
                    bbox = None,
                    block_type = "text",
                    source_index = 0,
                )
            ],
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(chunk.metadata["page_numbers"] == [3] for chunk in chunks)
        )

    def test_empty_pdf_source_blocks_use_the_existing_generic_path(self) -> None:
        document = Document(
            page_content = "# Heading\n\nGeneric content.",
            metadata = self.document.metadata,
        )
        module = DataPreparationModule()

        generic = module.prepare_documents([document])
        fallback = module.prepare_pdf_document(document, [])

        self.assertEqual(
            [(chunk.page_content, chunk.metadata) for chunk in fallback],
            [(chunk.page_content, chunk.metadata) for chunk in generic],
        )
        self.assertNotIn("page_numbers", fallback[0].metadata)

    def test_invalid_content_list_is_ignored(self) -> None:
        self.assertIsNone(
            MinerUClient.extract_content_list(
                {"paper_content_list.json": b"not json"}
            )
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import unittest
from uuid import uuid4

from app.config import settings
from app.infrastructure.mineru.mineru_client import MinerUClient
from app.rag.data_preparation.data_preparation import DataPreparationModule
from app.rag.file_parser.file_parser import FileParser
from app.rag.index_construction.base import close_index_resources
from app.rag.index_construction.paper_content_index_construction import (
    PaperContentIndexConstructionModule,
)
from app.rag.index_construction.paper_index_construction import (
    PaperIndexConstructionModule,
)
from app.rag.processing.paper_rag_processor import (
    PaperRagInput,
    PaperRagProcessor,
)


# 将你提供的真实论文填入这里。
PAPERS: list[PaperRagInput] = [
    PaperRagInput(
         paper_id = 120,
         title = "Ares: An automated evaluation framework for retrieval-augmented generation systems",
         abstract = "Evaluating retrieval-augmented generation (RAG) systems traditionally relies on hand annotations for input queries, passages to re- trieve, and responses to generate. We intro- duce ARES, an Automated RAG Evaluation System, for evaluating RAG systems along the dimensions of context relevance, answer faithfulness, and answer relevance. By cre- ating its own synthetic training data, ARES finetunes lightweight LM judges to assess the quality of individual RAG components. To mitigate potential prediction errors, ARES uti- lizes a small set of human-annotated datapoints for prediction-powered inference (PPI). Across eight different knowledge-intensive tasks in KILT, SuperGLUE, and AIS, ARES accurately evaluates RAG systems while using only a few hundred human annotations during evaluation. Furthermore, ARES judges remain effective across domain shifts, proving accurate even after changing the type of queries and/or docu- ments used in the evaluated RAG systems. We make our code and datasets publicly available on Github.",
         pdf_url = "https://aclanthology.org/2024.naacl-long.20.pdf",
         year = 2024,
     ),

     PaperRagInput(
         paper_id = 121,
         title = "Evaluation of retrieval-augmented generation: A survey",
         abstract = "Retrieval-Augmented Generation (RAG) has recently gained traction in natural language processing. Numerous studies and real-world applications are leveraging its ability to enhance generative models through external informa- tion retrieval. Evaluating these RAG systems, however, poses unique challenges due to their hybrid structure and reliance on dynamic knowledge sources. To better understand these challenges, we conduct A Unified Evaluation Process of RAG (Auepora) and aim to provide a comprehensive overview of the evaluation and benchmarks of RAG systems. Specifically, we examine and compare several quantifiable metrics of the Retrieval and Generation components, such as rele- vance, accuracy, and faithfulness, within the current RAG benchmarks, encom- passing the possible output and ground truth pairs. We then analyze the various datasets and metrics, discuss the limitations of current benchmarks, and suggest potential directions to advance the field of RAG benchmarks.",
         pdf_url = "https://arxiv.org/pdf/2405.07437",
         year = 2024,
     ),
]


@unittest.skipUnless(
    os.getenv("RUN_RAG_INTEGRATION_TESTS") == "1",
    "Set RUN_RAG_INTEGRATION_TESTS = 1 to run MinerU/Qdrant integration tests",
)
class PaperRagProcessorRealTests(
    unittest.IsolatedAsyncioTestCase,
):
    async def asyncSetUp(self) -> None:
        if not PAPERS:
            self.skipTest("Add real papers to PAPERS before running")

        run_id = uuid4().hex[:8]
        self.paper_collection = f"paper_rag_real_{run_id}"
        self.content_collection = f"paper_content_rag_real_{run_id}"

        self.mineru_client = MinerUClient(
            token = settings.MINERU_TOKEN,
            base_url = settings.MINERU_BASE_URL,
            request_timeout = 60,
            parse_timeout = 900,
            poll_interval = 3,
        )
        self.file_parser = FileParser(
            mineru_client = self.mineru_client,
            max_parallel_batches = settings.MINERU_MAX_PARALLEL_BATCHES,
        )
        self.paper_index = PaperIndexConstructionModule(
            collection_name = self.paper_collection,
        )
        self.content_index = PaperContentIndexConstructionModule(
            collection_name = self.content_collection,
        )
        self.processor = PaperRagProcessor(
            file_parser = self.file_parser,
            data_preparation = DataPreparationModule(),
            paper_index = self.paper_index,
            paper_content_index = self.content_index,
        )

    async def asyncTearDown(self) -> None:
        try:
            client = self.paper_index.client

            for collection_name in (
                self.paper_collection,
                self.content_collection,
            ):
                if await client.collection_exists(
                    collection_name = collection_name,
                ):
                    await client.delete_collection(
                        collection_name = collection_name,
                    )
        finally:
            await self.mineru_client.close()
            await close_index_resources()

    async def test_indexes_real_papers(self) -> None:
        results = await self.processor.process_many(PAPERS)

        for result in results:
            print(
                f"paper_id = {result.paper_id}, "
                f"status = {result.status}, "
                f"chunks = {result.chunk_count}, "
                f"error = {result.error}"
            )

        failed = [
            result
            for result in results
            if result.status == "failed"
        ]
        self.assertEqual(
            failed,
            [],
            f"Unexpected failed papers: {failed}",
        )

        ready = [
            result
            for result in results
            if result.status == "ready"
        ]
        self.assertEqual(
            len(ready),
            len(PAPERS),
            "Every test paper should produce indexable content",
        )

        client = self.paper_index.client

        paper_count = await client.count(
            collection_name = self.paper_collection,
            exact = True,
        )
        content_count = await client.count(
            collection_name = self.content_collection,
            exact = True,
        )

        self.assertEqual(paper_count.count, len(PAPERS))
        self.assertEqual(
            content_count.count,
            sum(result.chunk_count for result in ready),
        )


if __name__ == "__main__":
    unittest.main()
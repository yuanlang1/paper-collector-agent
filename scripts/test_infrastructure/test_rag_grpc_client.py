import unittest
from unittest.mock import AsyncMock

from app.infrastructure.grpc.rag_grpc_client import RagGrpcClient
from app.protos.rag.v1 import rag_pb2


BATCH_ID = "cde4e7fa-6eee-44b3-af4b-c2d7972d5d4b"


class RagGrpcClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_maps_empty_claim_as_success(self) -> None:
        response = rag_pb2.ClaimTaskRagPapersResponse(
            code=0,
            success=True,
            task_id=38,
            batch_id=BATCH_ID,
            claimed_count=0,
            task_state=8,
        )
        stub = type(
            "Stub",
            (),
            {"ClaimTaskRagPapers": AsyncMock(return_value=response)},
        )()
        client = RagGrpcClient()
        client._get_stub = AsyncMock(return_value=stub)

        result = await client.claim_task_rag_papers(
            task_id=38,
            batch_id=BATCH_ID,
            limit=20,
            lease_seconds=1800,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["claimed_count"], 0)
        request = stub.ClaimTaskRagPapers.await_args.args[0]
        self.assertEqual(request.task_id, 38)
        self.assertEqual(request.batch_id, BATCH_ID)

    async def test_complete_maps_results_and_progress(self) -> None:
        response = rag_pb2.CompleteTaskRagBatchResponse(
            code=0,
            success=True,
            task_id=38,
            batch_id=BATCH_ID,
            task_state=8,
            total_count=2,
            ready_count=1,
            failed_count=1,
            progress_percent=100,
        )
        stub = type(
            "Stub",
            (),
            {"CompleteTaskRagBatch": AsyncMock(return_value=response)},
        )()
        client = RagGrpcClient()
        client._get_stub = AsyncMock(return_value=stub)

        result = await client.complete_task_rag_batch(
            task_id=38,
            batch_id=BATCH_ID,
            results=[
                {"paper_id": 101, "status": "ready", "chunk_count": 16},
                {
                    "paper_id": 102,
                    "status": "failed",
                    "error": "Qdrant upsert failed",
                },
            ],
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["failed_count"], 1)
        request = stub.CompleteTaskRagBatch.await_args.args[0]
        self.assertEqual(request.results[0].status, rag_pb2.RAG_BATCH_RESULT_READY)
        self.assertEqual(request.results[1].status, rag_pb2.RAG_BATCH_RESULT_FAILED)

    async def test_rejects_invalid_result_before_service_call(self) -> None:
        client = RagGrpcClient()
        client._get_stub = AsyncMock()

        result = await client.complete_task_rag_batch(
            task_id=38,
            batch_id=BATCH_ID,
            results=[{"paper_id": 101, "status": "skipped"}],
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "skipped result requires an error")
        client._get_stub.assert_not_awaited()

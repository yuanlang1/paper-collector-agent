import unittest
from unittest.mock import AsyncMock

from app.infrastructure.grpc.task_paper_relation_grpc_client import (
    TaskPaperRelationGrpcClient,
)
from app.protos.task_paper_relation.v1 import task_paper_relation_pb2


class TaskPaperRelationGrpcClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_unindexed_paper_rag_infos_maps_response(self) -> None:
        response = task_paper_relation_pb2.GetUnindexedPaperRagInfosByTaskIdResponse(
            code=200,
            success=True,
            task_id=12,
            paper_infos=[
                task_paper_relation_pb2.PaperRagInfoItem(
                    paper_id=34,
                    title="A Paper",
                    url="https://example.test/paper",
                    paper_abstract="Abstract",
                )
            ],
        )
        stub = type(
            "Stub",
            (),
            {"GetUnindexedPaperRagInfosByTaskId": AsyncMock(return_value=response)},
        )()
        client = TaskPaperRelationGrpcClient()
        client._get_stub = AsyncMock(return_value=stub)

        result = await client.get_unindexed_paper_rag_infos_by_task_id(12)

        self.assertEqual(
            result["result"],
            {
                "task_id": 12,
                "paper_infos": [
                    {
                        "paper_id": 34,
                        "title": "A Paper",
                        "url": "https://example.test/paper",
                        "paper_abstract": "Abstract",
                    }
                ],
            },
        )
        self.assertTrue(result["ok"])
        request = stub.GetUnindexedPaperRagInfosByTaskId.await_args.args[0]
        self.assertEqual(request.task_id, 12)

    async def test_get_paper_ids_returns_service_failure(self) -> None:
        response = task_paper_relation_pb2.GetPaperIdsByTaskIdResponse(
            code=404,
            success=False,
            message="task not found",
        )
        stub = type(
            "Stub",
            (),
            {"GetPaperIdsByTaskId": AsyncMock(return_value=response)},
        )()
        client = TaskPaperRelationGrpcClient()
        client._get_stub = AsyncMock(return_value=stub)

        result = await client.get_paper_ids_by_task_id(12)

        self.assertEqual(result["error"], "task not found")
        self.assertEqual(result["metadata"]["code"], 404)
        self.assertFalse(result["ok"])

    async def test_rejects_invalid_task_id_before_calling_service(self) -> None:
        client = TaskPaperRelationGrpcClient()
        client._get_stub = AsyncMock()

        result = await client.get_paper_ids_by_task_id(0)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "task_id must be a positive integer")
        client._get_stub.assert_not_awaited()

import unittest

from app.infrastructure.grpc.task_service_grpc_client import (
    TaskServiceGrpcClient,
    TaskState,
)
from app.llm.graph.workflows.paper_search.nodes.update_task_status import (
    UpdatePaperSearchTaskStatusNode,
)
from app.llm.graph.workflows.paper_search.nodes.create_task import (
    CreatePaperSearchTaskNode,
)
from app.llm.subagents.paper_search.contracts import (
    build_paper_search_handoff,
)
from app.protos.task.v1 import task_pb2


class _Stub:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def UpdateTaskStatus(self, request, timeout):
        self.requests.append((request, timeout))
        return self.response


class _Client(TaskServiceGrpcClient):
    def __init__(self, stub):
        self.stub = stub

    async def _get_stub(self):
        return self.stub


class _RecordingClient:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {
            "ok": True,
            "result": {"updated": True},
            "error": None,
        }

    async def update_task_status(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _RecordingRagRunner:
    def __init__(self) -> None:
        self.task_ids: list[int] = []

    async def notify(self, task_id: int) -> bool:
        self.task_ids.append(task_id)
        return True


class _CreateTaskClient:
    def __init__(self):
        self.create_requests = []
        self.status_calls = []

    async def add_query_task(self, request):
        self.create_requests.append(request)
        return {
            "ok": True,
            "result": {"task_id": 38},
            "error": None,
        }

    async def update_task_status(self, **kwargs):
        self.status_calls.append(kwargs)
        return {
            "ok": True,
            "result": {"updated": True},
            "error": None,
        }


class TaskStatusTests(unittest.IsolatedAsyncioTestCase):
    def test_task_state_values_match_service_contract(self) -> None:
        self.assertEqual(
            {
                state.name: int(state)
                for state in TaskState
            },
            {
                "SEARCH_PENDING": 0,
                "SEARCH_RUNNING": 1,
                "SEARCH_COMPLETED": 2,
                "SEARCH_FAILED": 3,
                "SEARCH_PARTIAL_COMPLETED": 4,
                "CANCELLED": 5,
                "RAG_RUNNING": 6,
                "RAG_FAILED": 7,
                "RAG_COMPLETED": 8,
            },
        )

    async def test_created_task_is_marked_search_running(self) -> None:
        client = _CreateTaskClient()
        result = await CreatePaperSearchTaskNode(client=client)(
            {
                "original_prompt": "search RAG papers",
                "search_tag": {
                    "yearTag": 0,
                    "paperTag": [1],
                    "sourceTag": ["arXiv"],
                },
                "query_understanding": {
                    "topic": "RAG",
                    "intent": "method",
                    "reasoning": "test",
                },
                "warnings": [],
                "progress": {},
            }
        )

        self.assertNotIn(
            "task_state",
            client.create_requests[0],
        )
        self.assertEqual(
            client.status_calls[0]["task_state"],
            TaskState.SEARCH_RUNNING,
        )
        self.assertEqual(
            result["remote_task_state"],
            "SEARCH_RUNNING",
        )

    async def test_client_sends_failed_status_and_error_message(self) -> None:
        stub = _Stub(
            task_pb2.UpdateTaskStatusResponse(
                code=0,
                success=True,
                updated=True,
            )
        )
        client = _Client(stub)

        result = await client.update_task_status(
            task_id=38,
            task_state=TaskState.SEARCH_FAILED,
            error_message="PDF download failed",
        )

        self.assertTrue(result["ok"])
        request, _timeout = stub.requests[0]
        self.assertEqual(request.task_id, 38)
        self.assertEqual(request.task_state, TaskState.SEARCH_FAILED)
        self.assertEqual(request.error_message, "PDF download failed")

    async def test_client_clears_error_for_non_failed_status(self) -> None:
        stub = _Stub(
            task_pb2.UpdateTaskStatusResponse(success=True, updated=True)
        )
        client = _Client(stub)

        await client.update_task_status(
            task_id=38,
            task_state=TaskState.SEARCH_COMPLETED,
            error_message="old error",
        )

        request, _timeout = stub.requests[0]
        self.assertEqual(request.error_message, "")

    async def test_terminal_node_maps_partial_failure(self) -> None:
        client = _RecordingClient()
        node = UpdatePaperSearchTaskStatusNode(client=client)

        result = await node(
            {
                "paper_service_task_id": 38,
                "stage": "partial_failed",
                "status": "partial_failed",
                "warnings": [],
                "progress": {},
            }
        )

        self.assertEqual(
            client.calls[0]["task_state"],
            TaskState.SEARCH_PARTIAL_COMPLETED,
        )
        self.assertEqual(client.calls[0]["error_message"], None)
        self.assertEqual(result["progress"]["task_status_updated"], 1)
        self.assertEqual(
            result["remote_task_state"],
            TaskState.SEARCH_PARTIAL_COMPLETED.name,
        )

    async def test_terminal_node_maps_completed_and_failed(self) -> None:
        cases = [
            (
                {
                    "stage": "completed",
                    "status": "completed",
                },
                TaskState.SEARCH_COMPLETED,
            ),
            (
                {
                    "stage": "failed",
                    "status": "failed",
                    "error": "search failed",
                },
                TaskState.SEARCH_FAILED,
            ),
        ]

        for state, expected in cases:
            client = _RecordingClient()
            rag_runner = _RecordingRagRunner()
            result = await UpdatePaperSearchTaskStatusNode(
                client=client,
                rag_runner=rag_runner,
            )(
                {
                    "paper_service_task_id": 38,
                    "warnings": [],
                    "progress": {},
                    **state,
                }
            )

            self.assertEqual(
                client.calls[0]["task_state"],
                expected,
            )
            self.assertEqual(
                result["remote_task_state"],
                expected.name,
            )
            self.assertEqual(
                rag_runner.task_ids,
                [38] if expected is TaskState.SEARCH_COMPLETED else [],
            )

    async def test_terminal_node_keeps_workflow_result_when_update_fails(self) -> None:
        client = _RecordingClient(
            {"ok": False, "result": None, "error": "task not found"}
        )
        node = UpdatePaperSearchTaskStatusNode(client=client)

        result = await node(
            {
                "paper_service_task_id": 38,
                "stage": "completed",
                "status": "completed",
                "warnings": [],
            }
        )

        self.assertEqual(result["task_status_update_error"], "task not found")
        self.assertIn("task not found", result["warnings"][0])
        self.assertEqual(result["stage"], "partial_failed")
        self.assertEqual(result["status"], "partial_failed")
        self.assertTrue(result["degraded"])
        self.assertEqual(result["progress"]["task_status_updated"], 0)
        self.assertIsNone(result["remote_task_state"])
        handoff = build_paper_search_handoff({
            "paper_service_task_id": 38,
            **result,
        })
        self.assertEqual(handoff.status, "partial")
        self.assertTrue(handoff.retryable)
        self.assertEqual(
            handoff.data["task_status_update_error"],
            "task not found",
        )

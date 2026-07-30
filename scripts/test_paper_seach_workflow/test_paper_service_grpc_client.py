import unittest

from app.infrastructure.paper_service_grpc_client import PaperServiceGrpcClient


class PaperServiceGrpcClientTests(unittest.TestCase):
    def test_build_request_allows_zero_venue_id(self) -> None:
        request = PaperServiceGrpcClient._build_request(
            {
                "client_key": "run-1:0",
                "paper_info": {
                    "title": "A Preprint",
                    "authors": "Ada Lovelace",
                    "venue_id": 0,
                },
            }
        )

        self.assertEqual(request.venue_id, 0)

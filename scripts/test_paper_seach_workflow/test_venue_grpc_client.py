import unittest

from app.infrastructure.venue_service_grpc_client import (
    VenueServiceGrpcClient,
)


class VenueServiceGrpcClientTests(unittest.TestCase):
    def test_optional_numeric_fields_are_omitted_when_none(self) -> None:
        request = VenueServiceGrpcClient._build_save_request(
            standard_name="Test Conference",
        )

        self.assertFalse(request.HasField("type"))
        self.assertFalse(request.HasField("sci_if"))

    def test_optional_numeric_fields_keep_explicit_zero(self) -> None:
        request = VenueServiceGrpcClient._build_save_request(
            standard_name="Test Conference",
            type=0,
            sci_if=0.0,
        )

        self.assertTrue(request.HasField("type"))
        self.assertTrue(request.HasField("sci_if"))
        self.assertEqual(request.type, 0)
        self.assertEqual(request.sci_if, 0.0)

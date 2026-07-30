import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.nodes.cleanup import (
    CleanupDownloadedPdfsNode,
)


class CleanupDownloadedPdfsNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_deletes_downloaded_pdfs_and_keeps_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            pdf_path = Path(directory) / "papers" / "run-1" / "paper.pdf"
            pdf_path.parent.mkdir(parents=True)
            pdf_path.write_bytes(b"%PDF-test")
            manifest_path = Path(directory) / "run-1" / "result.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("{}", encoding="utf-8")

            result = await CleanupDownloadedPdfsNode(store)(
                {
                    "downloaded_pdf_paths": [str(pdf_path)],
                    "progress": {"persistence_saved": 1},
                }
            )

            self.assertFalse(pdf_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(result["downloaded_pdf_paths"], [])
            self.assertEqual(result["progress"]["pdf_files_deleted"], 1)
            self.assertEqual(result["progress"]["persistence_saved"], 1)

    async def test_cleanup_failure_degrades_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            node = CleanupDownloadedPdfsNode(store)

            with patch.object(
                node,
                "_delete",
                side_effect=OSError("file is locked"),
            ):
                result = await node(
                    {
                        "stage": "completed",
                        "status": "completed",
                        "downloaded_pdf_paths": ["paper.pdf"],
                        "warnings": [],
                        "progress": {},
                    }
                )

        self.assertEqual(result["stage"], "partial_failed")
        self.assertEqual(result["status"], "partial_failed")
        self.assertTrue(result["degraded"])
        self.assertEqual(result["pdf_cleanup_error"], "file is locked")


if __name__ == "__main__":
    unittest.main()

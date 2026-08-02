from __future__ import annotations

import unittest
from pathlib import Path

from app.config import settings
from app.infrastructure.mineru.mineru_client import MinerUClient


TEST_FILE_ID = "mineru-real-test"

TEST_FILE_URI = "https://aclanthology.org/2024.naacl-long.20.pdf"

OUTPUT_DIR = Path("scripts/file_parser/output")


class MinerURealTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = MinerUClient(
            token=settings.MINERU_TOKEN,
            base_url=getattr(
                settings,
                "MINERU_BASE_URL",
                "https://mineru.net",
            ),
            request_timeout=60,
            parse_timeout=900,
            poll_interval=3,
        )

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_parse_real_file(self) -> None:
        task_id = await self.client.submit_single_url(
            uri=TEST_FILE_URI,
            data_id=TEST_FILE_ID,
            model_version="vlm",
            options={
                "is_ocr": False,
                "enable_formula": True,
                "enable_table": True,
                "language": "en",
                "no_cache": True,
            },
        )

        print(f"\nMinerU task submitted: {task_id}")

        result = await self.client.wait_single(task_id)

        self.assertEqual(result["state"], "done")
        self.assertIn("full_zip_url", result)

        print(
            "MinerU task completed: "
            f"{result['full_zip_url']}"
        )

        zip_content = await self.client.download_result_zip(
            result["full_zip_url"]
        )

        self.assertGreater(len(zip_content), 0)

        result_files = self.client.extract_result_files(
            zip_content
        )

        self.assertGreater(len(result_files), 0)

        markdown = self.client.extract_markdown(
            result_files
        )

        self.assertTrue(markdown.strip())

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        markdown_path = (
            OUTPUT_DIR
            / f"{TEST_FILE_ID}.md"
        )

        markdown_path.write_text(
            markdown,
            encoding="utf-8",
        )

        zip_path = (
            OUTPUT_DIR
            / f"{TEST_FILE_ID}.zip"
        )

        zip_path.write_bytes(zip_content)

        print(f"Markdown length: {len(markdown)}")
        print(f"ZIP size: {len(zip_content)} bytes")
        print(f"Result file count: {len(result_files)}")
        print(f"Markdown saved to: {markdown_path.resolve()}")
        print(f"ZIP saved to: {zip_path.resolve()}")

        self.assertTrue(markdown_path.exists())
        self.assertTrue(zip_path.exists())


if __name__ == "__main__":
    unittest.main()
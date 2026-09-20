import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_test_videos.py"


class ExportTestVideosTests(unittest.TestCase):
    def test_exports_only_unique_test_videos_with_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            first = videos / "aerobic_gymnastics" / "routine.mp4"
            second = videos / "basketball" / "shot.mkv"
            ignored = videos / "training" / "ignored.mp4"
            for path in (first, second, ignored):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.name.encode())

            metadata = root / "test.jsonl"
            metadata.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {"video_id": "aerobic_gymnastics/routine"},
                        {"video_id": "aerobic_gymnastics/routine"},
                        {"video_id": "basketball/shot.mkv", "split": "test"},
                        {"video_id": "training/ignored", "split": "train"},
                    )
                ),
                encoding="utf-8",
            )
            output = root / "test_videos"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--metadata-path",
                    str(metadata),
                    "--video-root",
                    str(videos),
                    "--output-root",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "aerobic_gymnastics" / "routine.mp4").is_file())
            self.assertTrue((output / "basketball" / "shot.mkv").is_file())
            self.assertFalse((output / "training" / "ignored.mp4").exists())
            manifest = json.loads((output / "subset_manifest.json").read_text())
            self.assertEqual(manifest["video_count"], 2)
            self.assertTrue(ignored.is_file())


if __name__ == "__main__":
    unittest.main()

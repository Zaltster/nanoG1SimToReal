import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / "scripts" / "mirror_latest_checkpoint.py"


class MirrorLatestCheckpointTest(unittest.TestCase):
    def run_mirror(self, checkpoint_dir: Path, output_dir: Path) -> None:
        subprocess.run(
            ["python3", str(MIRROR), str(checkpoint_dir), str(output_dir)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write_checkpoint(self, checkpoint_dir: Path, counter: int) -> None:
        run_dir = checkpoint_dir / "g1gpu" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / f"{counter:016d}.bin").write_bytes(b"checkpoint")

    def test_removes_eval_from_different_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = root / "checkpoints"
            out = root / "outputs"
            out.mkdir()
            self.write_checkpoint(checkpoints, 10)
            (out / "run_metadata.json").write_text('{"time_unix": 123}\n')
            (out / "checkpoint_eval_latest.json").write_text('{"counter": 5, "run_started_unix": 99}\n')
            (out / "checkpoint_eval.jsonl").write_text("{}\n")

            self.run_mirror(checkpoints, out)

            self.assertFalse((out / "checkpoint_eval_latest.json").exists())
            self.assertFalse((out / "checkpoint_eval.jsonl").exists())

    def test_removes_eval_ahead_of_current_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = root / "checkpoints"
            out = root / "outputs"
            out.mkdir()
            self.write_checkpoint(checkpoints, 10)
            (out / "run_metadata.json").write_text('{"time_unix": 123}\n')
            (out / "checkpoint_eval_latest.json").write_text('{"counter": 99, "run_started_unix": 123}\n')
            (out / "checkpoint_eval.jsonl").write_text("{}\n")

            self.run_mirror(checkpoints, out)

            self.assertFalse((out / "checkpoint_eval_latest.json").exists())
            self.assertFalse((out / "checkpoint_eval.jsonl").exists())

    def test_keeps_eval_for_current_run_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = root / "checkpoints"
            out = root / "outputs"
            out.mkdir()
            self.write_checkpoint(checkpoints, 10)
            (out / "run_metadata.json").write_text('{"time_unix": 123}\n')
            (out / "checkpoint_eval_latest.json").write_text('{"counter": 5, "run_started_unix": 123}\n')
            (out / "checkpoint_eval.jsonl").write_text("{}\n")

            self.run_mirror(checkpoints, out)

            self.assertTrue((out / "checkpoint_eval_latest.json").exists())
            self.assertTrue((out / "checkpoint_eval.jsonl").exists())
            latest = json.loads((out / "latest_checkpoint.json").read_text())
            self.assertEqual(latest["counter"], 10)


if __name__ == "__main__":
    unittest.main()

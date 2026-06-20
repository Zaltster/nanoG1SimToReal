"""Print the CUDA/Python facts that matter before local nanoG1 training."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=20).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def discover(name: str, patterns: list[str]) -> tuple[str | None, list[str]]:
    found = shutil.which(name)
    hits: list[str] = []
    for pattern in patterns:
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            if path.is_file():
                hits.append(str(path))
    return found or (hits[0] if hits else None), hits


def main() -> None:
    nvcc, nvcc_candidates = discover(
        "nvcc",
        ["/usr/local/cuda*/bin/nvcc", "/usr/local/cuda*/targets/*/bin/nvcc", "/usr/bin/nvcc"],
    )
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "nvidia_smi": shutil.which("nvidia-smi"),
        "nvcc": nvcc,
        "nvcc_candidates": nvcc_candidates,
        "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]),
        "nvcc_version": run([nvcc, "--version"]) if nvcc else "unavailable",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["torch_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_error"] = repr(exc)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

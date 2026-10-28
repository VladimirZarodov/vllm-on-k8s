#!/usr/bin/env python3
"""Download a Hugging Face model into a Kubernetes PVC (initContainer entrypoint).

The script is deliberately conservative:

- the HF token is read only from the ``HF_TOKEN`` environment variable
  (injected by a Kubernetes Secret) and is never printed;
- the model is marked as ready only after the weights have been downloaded
  and validated;
- when the model is already complete, the script exits without any network
  call, so repeated pod restarts do not touch Hugging Face again;
- any failure leaves the model *without* the ready marker and exits non-zero,
  so Kubernetes retries the initContainer.

Environment variables (provided by ConfigMap ``vllm-config`` and Secret
``vllm-secrets``):

    MODEL_REPO  Hugging Face repo id (default Qwen/Qwen2.5-0.5B-Instruct)
    MODEL_DIR   Directory on the mounted PVC (default /models/model)
    HF_TOKEN    Optional token for gated repos; empty means anonymous
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MARKER_NAME = ".model_ready.json"
MARKER_VERSION = 1

DEFAULT_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_DIR = "/models/model"

MIN_WEIGHTS_BYTES = 50 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 15


def _scrub(text: str) -> str:
    """Remove the token value from any diagnostic text, just in case."""
    token = os.environ.get("HF_TOKEN") or ""
    if token and token in text:
        return text.replace(token, "<redacted>")
    return text


def log(message: str) -> None:
    print(f"[model-init] {_scrub(message)}", flush=True)


def model_dir_path() -> Path:
    return Path(os.environ.get("MODEL_DIR") or DEFAULT_MODEL_DIR)


def repo_id() -> str:
    return (os.environ.get("MODEL_REPO") or DEFAULT_REPO).strip() or DEFAULT_REPO


def read_marker(marker_path: Path) -> dict | None:
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def list_model_files(directory: Path) -> dict[str, int]:
    """Name -> size for real model files (caches and the marker excluded)."""
    files: dict[str, int] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(directory).as_posix()
        if rel == MARKER_NAME or rel.startswith(".cache/"):
            continue
        files[rel] = path.stat().st_size
    return files


def validate_model_files(directory: Path) -> list[str]:
    """Return a list of human-readable problems; empty list means complete."""
    problems: list[str] = []

    if not directory.is_dir():
        return [f"model directory {directory} does not exist"]

    # 1. Hugging Face config must exist and look like a model config.
    config_path = directory / "config.json"
    config: object = {}
    if not config_path.is_file() or config_path.stat().st_size == 0:
        problems.append("config.json is missing or empty")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"config.json cannot be parsed: {type(exc).__name__}")
    architectures = config.get("architectures") if isinstance(config, dict) else None
    if not isinstance(architectures, list) or not architectures:
        problems.append("config.json has no non-empty 'architectures' list")

    # 2. Every weight shard referenced by the index must be present.
    weight_files: list[Path] = []
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index: object = {}
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"model.safetensors.index.json cannot be parsed: {type(exc).__name__}")
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if isinstance(weight_map, dict) and weight_map:
            for name in sorted(set(weight_map.values())):
                shard = directory / name
                if not shard.is_file() or shard.stat().st_size == 0:
                    problems.append(f"weight shard is missing or empty: {name}")
                else:
                    weight_files.append(shard)

    # 3. Otherwise expect at least one plain weights file.
    if not weight_files:
        found = sorted(
            path
            for path in [*directory.glob("*.safetensors"), *directory.glob("*.bin")]
            if path.is_file() and path.stat().st_size > 0
        )
        if found:
            weight_files = found
        elif not problems:
            problems.append("no model weight files (*.safetensors / *.bin) found")

    # 4. Sanity check: tiny weights mean a truncated download.
    if weight_files:
        total_bytes = sum(path.stat().st_size for path in weight_files)
        if total_bytes < MIN_WEIGHTS_BYTES:
            problems.append(
                f"weights look truncated: {total_bytes / (1024 * 1024):.1f} MiB "
                f"across {len(weight_files)} file(s)"
            )

    # 5. Tokenizer artifacts required by vLLM for this model family.
    if not (directory / "tokenizer.json").is_file() and not (directory / "vocab.json").is_file():
        problems.append("no tokenizer file (tokenizer.json / vocab.json) found")
    if not (directory / "tokenizer_config.json").is_file():
        problems.append("tokenizer_config.json is missing")

    return problems


def download_model(directory: Path, repo: str) -> bool:
    """Try snapshot_download a few times. True if a download call completed."""
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            log(f"download attempt {attempt}/{DOWNLOAD_ATTEMPTS}: {repo} -> {directory}")
            snapshot_download(repo_id=repo, local_dir=str(directory), token=token)
            log("download call finished, validating files")
            return True
        except Exception as exc:  # noqa: BLE001 - hub errors are heterogeneous
            log(f"attempt {attempt} failed: {type(exc).__name__}: {_scrub(str(exc))[:2000]}")
            if attempt < DOWNLOAD_ATTEMPTS:
                delay = RETRY_DELAY_SECONDS * attempt
                log(f"retrying in {delay}s")
                time.sleep(delay)
    return False


def write_marker(marker_path: Path, repo: str, files: dict[str, int]) -> None:
    marker_doc = {
        "version": MARKER_VERSION,
        "repo": repo,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_count": len(files),
        "total_bytes": sum(files.values()),
        "files": files,
    }
    tmp_path = marker_path.with_name(marker_path.name + ".tmp")
    tmp_path.write_text(json.dumps(marker_doc, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, marker_path)  # atomic: marker appears only when complete


def main() -> int:
    directory = model_dir_path()
    repo = repo_id()

    log(f"model repo: {repo}")
    log(f"model dir:  {directory}")
    if os.environ.get("HF_TOKEN"):
        log("HF_TOKEN is present (the value is never printed)")
    else:
        log("HF_TOKEN is empty; public repos are fetched anonymously")

    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        log(f"ERROR: PVC is not writable under {directory}: {type(exc).__name__}")
        log("hint: check pod securityContext fsGroup and volume mountPath")
        return 1

    marker_path = directory / MARKER_NAME
    marker = read_marker(marker_path)
    if marker is not None:
        marker_repo = marker.get("repo")
        if marker_repo != repo:
            log(f"ERROR: marker records repo '{marker_repo}' but MODEL_REPO is '{repo}'.")
            log("refusing to mix models in one directory: wipe the PVC or change MODEL_DIR")
            return 1
        problems = validate_model_files(directory)
        if not problems:
            log(f"model '{repo}' is already complete (marker + files valid)")
            log("skipping download: no network access required")
            return 0
        log("marker exists but files are invalid/incomplete, re-downloading:")
        for problem in problems:
            log(f"  - {problem}")
    else:
        log("no ready marker found, starting first download")

    if not download_model(directory, repo):
        log("ERROR: download FAILED; model is NOT marked as ready")
        return 1

    problems = validate_model_files(directory)
    if problems:
        log("ERROR: download finished but the model is incomplete:")
        for problem in problems:
            log(f"  - {problem}")
        log("marker is NOT written; the initContainer will retry")
        return 1

    files = list_model_files(directory)
    write_marker(marker_path, repo, files)
    total_mib = sum(files.values()) / (1024 * 1024)
    log(f"model ready: {len(files)} file(s), {total_mib:.1f} MiB total")
    log(f"completion marker written: {marker_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

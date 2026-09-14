#!/usr/bin/env python3
"""Submit deterministic texture jobs to a local ComfyUI server.

The workflow must be saved in ComfyUI's API format.  A job can override any
node input with a ``NODE_ID.INPUT_NAME`` key, which keeps this client usable
with different checkpoints and custom nodes without baking model names into
the repository.

Example:

    .venv/bin/python tools/comfyui_texture_batch.py \
        --workflow tools/comfyui_building_texture_api.json \
        --jobs tools/comfyui_building_texture_jobs.json \
        --output-dir /tmp/ortho4xp-building-textures
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


class ComfyUIError(RuntimeError):
    """Raised when a ComfyUI job cannot be submitted or completed."""


def set_node_input(workflow: dict[str, Any], path: str, value: Any) -> None:
    """Set ``node_id.input_name`` in an API-format workflow."""
    try:
        node_id, input_name = path.split(".", 1)
        node = workflow[str(node_id)]
        inputs = node["inputs"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid workflow input path: {path}") from exc
    inputs[input_name] = value


def apply_job(
    workflow: dict[str, Any],
    job: dict[str, Any],
    *,
    seed_node: str | None = "6",
    seed_input: str = "seed",
    filename_node: str | None = "8",
    filename_input: str = "filename_prefix",
) -> dict[str, Any]:
    """Return one workflow copy with job-specific inputs applied."""
    result = copy.deepcopy(workflow)
    for path, value in job.get("overrides", {}).items():
        set_node_input(result, str(path), value)
    if seed_node is not None and "seed" in job:
        set_node_input(result, f"{seed_node}.{seed_input}", int(job["seed"]))
    if filename_node is not None and job.get("name"):
        set_node_input(result, f"{filename_node}.{filename_input}", str(job["name"]))
    return result


def _json_request(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # urllib has several platform-specific errors.
        raise ComfyUIError(f"ComfyUI request failed: {url}: {exc}") from exc


def submit_workflow(base_url: str, workflow: dict[str, Any], client_id: str, timeout: float = 30.0) -> str:
    response = _json_request(
        urljoin(base_url.rstrip("/") + "/", "prompt"),
        {"prompt": workflow, "client_id": client_id},
        timeout,
    )
    if response.get("error") or not response.get("prompt_id"):
        raise ComfyUIError(f"ComfyUI rejected workflow: {json.dumps(response, ensure_ascii=False)}")
    return str(response["prompt_id"])


def _history_entry(payload: Any, prompt_id: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    entry = payload.get(prompt_id)
    return entry if isinstance(entry, dict) else None


def first_image(entry: dict[str, Any]) -> dict[str, str]:
    """Return the first image metadata emitted by any output node."""
    outputs = entry.get("outputs", {})
    if isinstance(outputs, dict):
        for output in outputs.values():
            for image in output.get("images", []) if isinstance(output, dict) else []:
                if isinstance(image, dict) and image.get("filename"):
                    return {key: str(image.get(key, "")) for key in ("filename", "subfolder", "type")}
    raise ComfyUIError("ComfyUI completed without an image output")


def wait_for_image(base_url: str, prompt_id: str, timeout_seconds: float, poll_seconds: float = 0.5) -> dict[str, str]:
    deadline = time.monotonic() + timeout_seconds
    history_url = urljoin(base_url.rstrip("/") + "/", f"history/{prompt_id}")
    while time.monotonic() < deadline:
        entry = _history_entry(_json_request(history_url), prompt_id)
        if entry:
            status = entry.get("status", {})
            if isinstance(status, dict) and status.get("status_str") in {"error", "failed"}:
                raise ComfyUIError(f"ComfyUI job failed: {json.dumps(status, ensure_ascii=False)}")
            try:
                return first_image(entry)
            except ComfyUIError:
                pass
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    raise ComfyUIError(f"timed out waiting for ComfyUI prompt {prompt_id}")


def download_image(base_url: str, metadata: dict[str, str], timeout: float = 30.0) -> bytes:
    query = urlencode({key: metadata.get(key, "") for key in ("filename", "subfolder", "type")})
    request = Request(urljoin(base_url.rstrip("/") + "/", f"view?{query}"), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise ComfyUIError(f"could not download ComfyUI output: {exc}") from exc


def _safe_name(value: str) -> str:
    name = Path(value).name
    if (
        not name
        or name != value
        or "/" in value
        or "\\" in value
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
    ):
        raise ValueError(f"invalid job name: {value}")
    return name


def load_jobs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = data.get("jobs") if isinstance(data, dict) else data
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs JSON must contain a non-empty list")
    result = []
    for job in jobs:
        if not isinstance(job, dict) or not job.get("name"):
            raise ValueError("each texture job requires a name")
        _safe_name(str(job["name"]))
        result.append(job)
    return result


def run_jobs(
    workflow_path: Path,
    jobs_path: Path,
    output_dir: Path,
    base_url: str,
    *,
    seed_node: str | None,
    filename_node: str | None,
    timeout_seconds: float,
    dry_run: bool,
) -> int:
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    jobs = load_jobs(jobs_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    client_id = str(uuid.uuid4())
    for job in jobs:
        name = _safe_name(str(job["name"]))
        job_workflow = apply_job(
            workflow,
            job,
            seed_node=seed_node,
            filename_node=filename_node,
        )
        if dry_run:
            (output_dir / f"{name}.workflow.json").write_text(
                json.dumps(job_workflow, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"prepared {name}")
            continue
        prompt_id = submit_workflow(base_url, job_workflow, client_id)
        metadata = wait_for_image(base_url, prompt_id, timeout_seconds)
        output_path = output_dir / f"{name}{Path(metadata['filename']).suffix or '.png'}"
        output_path.write_bytes(download_image(base_url, metadata))
        print(f"saved {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--seed-node", default="6", help="workflow node ID containing the seed; use empty to disable")
    parser.add_argument("--filename-node", default="8", help="SaveImage node ID; use empty to disable")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return run_jobs(
        args.workflow,
        args.jobs,
        args.output_dir,
        args.url,
        seed_node=args.seed_node or None,
        filename_node=args.filename_node or None,
        timeout_seconds=args.timeout,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

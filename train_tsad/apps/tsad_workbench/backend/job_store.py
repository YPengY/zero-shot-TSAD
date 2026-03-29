from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .environment import JOB_LOG_LIMIT, PREVIEW_CACHE_LIMIT


@dataclass(slots=True)
class JobState:
    """Mutable state for one asynchronous workbench job."""

    job_id: str
    kind: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    logs: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


def read_json_artifact(path: Path) -> Any | None:
    """Read one optional JSON artifact without raising on transient failures."""

    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class JobStore:
    """Thread-safe store for background workbench jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create_job(self, kind: str) -> JobState:
        """Create and register one pending job."""

        job = JobState(job_id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> JobState | None:
        """Return one job by id."""

        with self._lock:
            return self._jobs.get(job_id)

    def append_log(self, job: JobState, line: str) -> None:
        """Append one log line to the job ring buffer."""

        text = line.rstrip()
        if not text:
            return
        with self._lock:
            job.logs.append(text)
            if len(job.logs) > JOB_LOG_LIMIT:
                del job.logs[:500]

    def set_artifacts(self, job: JobState, artifacts: dict[str, Any]) -> None:
        """Replace the current job artifacts."""

        with self._lock:
            job.artifacts = dict(artifacts)

    def to_payload(self, job: JobState) -> dict[str, Any]:
        """Serialize the job as an API response payload."""

        progress = None
        progress_path = job.artifacts.get("progress_path")
        if isinstance(progress_path, str):
            progress = read_json_artifact(Path(progress_path))
        return {
            "job_id": job.job_id,
            "kind": job.kind,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "logs": list(job.logs),
            "artifacts": dict(job.artifacts),
            "progress": progress,
            "result": job.result,
            "error": job.error,
        }

    def run_job(self, job: JobState, fn: Callable[[dict[str, Any], JobState], dict[str, Any]], payload: dict[str, Any]) -> None:
        """Execute one job function and persist its terminal state."""

        with self._lock:
            job.status = "running"
            job.started_at = time.time()
        try:
            result = fn(payload, job)
            with self._lock:
                job.status = "completed"
                job.result = result
                job.finished_at = time.time()
        except Exception as exc:  # pragma: no cover - defensive async guard
            self.append_log(job, traceback.format_exc())
            with self._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = time.time()


class PreviewStore:
    """Thread-safe LRU cache for in-memory preview payloads."""

    def __init__(self) -> None:
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, preview: dict[str, Any]) -> str:
        """Cache one preview payload and return its id."""

        preview_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._cache[preview_id] = preview
            self._cache.move_to_end(preview_id)
            while len(self._cache) > PREVIEW_CACHE_LIMIT:
                self._cache.popitem(last=False)
        return preview_id

    def get(self, preview_id: str) -> dict[str, Any] | None:
        """Return one cached preview and refresh its LRU position."""

        with self._lock:
            preview = self._cache.get(preview_id)
            if preview is None:
                return None
            self._cache.move_to_end(preview_id)
            return preview

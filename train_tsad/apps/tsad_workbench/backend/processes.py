from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .job_store import JobState, JobStore


def run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    job: JobState,
    job_store: JobStore,
    log_prefix: str | None = None,
) -> None:
    """Run one subprocess and stream its output into the job log."""

    prefix = f"[{log_prefix}] " if log_prefix else ""
    job_store.append_log(job, f"{prefix}$ {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if text:
            job_store.append_log(job, f"{prefix}{text}")
    rc = process.wait()
    if rc != 0:
        raise RuntimeError(f"{prefix}Command failed with exit code {rc}: {' '.join(cmd)}")


def run_parallel_split_generation(
    split_commands: list[tuple[str, list[str]]],
    *,
    cwd: Path,
    job: JobState,
    job_store: JobStore,
) -> None:
    """Run generation commands for multiple splits in parallel."""

    if not split_commands:
        return

    job_store.append_log(
        job, f"Launching {len(split_commands)} split generation processes in parallel."
    )
    failures: list[str] = []
    max_workers = min(len(split_commands), 3)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="wb-generate") as executor:
        future_to_split = {
            executor.submit(
                run_subprocess,
                cmd,
                cwd=cwd,
                job=job,
                job_store=job_store,
                log_prefix=f"split:{split}",
            ): split
            for split, cmd in split_commands
        }
        for future in as_completed(future_to_split):
            split = future_to_split[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{split}: {exc}")
            else:
                job_store.append_log(job, f"[split:{split}] generation completed.")

    if failures:
        raise RuntimeError("Split generation failed: " + "; ".join(failures))

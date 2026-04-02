from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from backend.dataset_browser import (
    build_sample_payload,
    build_samples_payload,
    build_window_payload,
)
from backend.environment import PREVIEW_CACHE_LIMIT, STATIC_DIR
from backend.job_services import run_generation_job, run_train_job
from backend.job_store import JobStore, PreviewStore
from backend.preview_service import (
    build_bootstrap_payload,
    build_preview_batch_payload,
    build_preview_payload,
)
from backend.runtime import build_run_info
from backend.studio_bridge import import_config_text, randomize_config
from backend.training_metrics import build_train_metrics_payload

JOB_STORE = JobStore()
PREVIEW_STORE = PreviewStore()


class WorkbenchRequestHandler(SimpleHTTPRequestHandler):
    """HTTP API + static asset handler for the TSAD workbench."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        """Dispatch GET endpoints or fall back to static file serving."""

        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/bootstrap":
            self._write_json(HTTPStatus.OK, build_bootstrap_payload())
            return
        if route == "/api/job":
            self._handle_get_job(parsed.query)
            return
        if route == "/api/run":
            self._handle_run_info(parsed.query)
            return
        if route == "/api/samples":
            self._handle_samples(parsed.query)
            return
        if route == "/api/sample":
            self._handle_sample(parsed.query)
            return
        if route == "/api/window":
            self._handle_window(parsed.query)
            return
        if route == "/api/preview-item":
            self._handle_preview_item(parsed.query)
            return
        if route == "/api/train-metrics":
            self._handle_train_metrics(parsed.query)
            return
        if route in {"/", "/index.html"}:
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        """Dispatch POST endpoints."""

        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/randomize":
            self._handle_randomize()
            return
        if route == "/api/preview":
            self._handle_preview()
            return
        if route == "/api/preview-batch":
            self._handle_preview_batch()
            return
        if route == "/api/import-config":
            self._handle_import_config()
            return
        if route == "/api/generate":
            self._handle_generate()
            return
        if route == "/api/train":
            self._handle_train()
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {route}"})

    def _handle_get_job(self, raw_query: str) -> None:
        query = parse_qs(raw_query)
        job_id = str(query.get("job_id", [""])[0]).strip()
        job = JOB_STORE.get_job(job_id)
        if job is None:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Job not found: {job_id}"})
            return
        self._write_json(HTTPStatus.OK, JOB_STORE.to_payload(job))

    def _handle_run_info(self, raw_query: str) -> None:
        try:
            query = parse_qs(raw_query)
            path_like = str(query.get("path", [""])[0]).strip()
            if not path_like:
                raise ValueError("path is required")
            self._write_json(HTTPStatus.OK, build_run_info(path_like))
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_randomize(self) -> None:
        try:
            body = self._read_json_body(optional=True)
            seed = (
                int(body["seed"])
                if isinstance(body, dict) and body.get("seed") is not None
                else None
            )
            self._write_json(HTTPStatus.OK, {"config": randomize_config(seed=seed)})
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_preview(self) -> None:
        try:
            body = self._read_json_body(optional=False)
            if not isinstance(body, dict) or "config" not in body:
                raise ValueError("Expected JSON object with 'config'.")
            payload = build_preview_payload(
                config=body["config"],
                seed_offset=int(body.get("seed_offset", 0)),
                cache=bool(body.get("cache", True)),
                preview_store=PREVIEW_STORE,
            )
            self._write_json(HTTPStatus.OK, payload)
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_preview_batch(self) -> None:
        try:
            body = self._read_json_body(optional=False)
            if not isinstance(body, dict) or "config" not in body:
                raise ValueError("Expected JSON object with 'config'.")
            count = max(1, min(int(body.get("count", 6)), PREVIEW_CACHE_LIMIT))
            seed_base = int(body.get("seed_base", body["config"].get("seed", 0) or 0))
            payload = build_preview_batch_payload(
                config=body["config"],
                count=count,
                seed_base=seed_base,
                preview_store=PREVIEW_STORE,
            )
            self._write_json(HTTPStatus.OK, payload)
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_preview_item(self, raw_query: str) -> None:
        query = parse_qs(raw_query)
        preview_id = str(query.get("preview_id", [""])[0]).strip()
        preview = PREVIEW_STORE.get(preview_id)
        if preview is None:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Preview not found: {preview_id}"})
            return
        self._write_json(HTTPStatus.OK, {"preview": preview, "preview_id": preview_id})

    def _handle_import_config(self) -> None:
        try:
            body = self._read_json_body(optional=False)
            if not isinstance(body, dict) or "text" not in body:
                raise ValueError("Expected JSON object with 'text'.")
            self._write_json(HTTPStatus.OK, import_config_text(str(body["text"])))
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_generate(self) -> None:
        try:
            payload = self._read_json_body(optional=False)
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object")
            self._start_job("generate", payload, run_generation_job)
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_train(self) -> None:
        try:
            payload = self._read_json_body(optional=False)
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object")
            self._start_job("train", payload, run_train_job)
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_samples(self, raw_query: str) -> None:
        try:
            query = parse_qs(raw_query)
            self._write_json(
                HTTPStatus.OK,
                build_samples_payload(
                    path_like=str(query.get("run_root", [""])[0]),
                    split=str(query.get("split", ["train"])[0]),
                    limit=max(1, min(int(query.get("limit", [300])[0]), 2000)),
                ),
            )
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_sample(self, raw_query: str) -> None:
        try:
            query = parse_qs(raw_query)
            self._write_json(
                HTTPStatus.OK,
                build_sample_payload(
                    path_like=str(query.get("run_root", [""])[0]),
                    split=str(query.get("split", ["train"])[0]),
                    sample_id=str(query.get("sample_id", [""])[0]),
                    feature_index=int(query.get("feature_index", [0])[0]),
                    slice_start=max(0, int(query.get("slice_start", [0])[0])),
                    slice_end=int(query.get("slice_end", [0])[0]),
                    context_size=max(1, int(query.get("context_size", [512])[0])),
                    stride=max(
                        1, int(query.get("stride", [int(query.get("context_size", [512])[0])])[0])
                    ),
                    patch_size=max(1, int(query.get("patch_size", [16])[0])),
                ),
            )
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_window(self, raw_query: str) -> None:
        try:
            query = parse_qs(raw_query)
            context_size = max(1, int(query.get("context_size", [512])[0]))
            self._write_json(
                HTTPStatus.OK,
                build_window_payload(
                    path_like=str(query.get("run_root", [""])[0]),
                    split=str(query.get("split", ["train"])[0]),
                    sample_id=str(query.get("sample_id", [""])[0]),
                    feature_index=int(query.get("feature_index", [0])[0]),
                    window_index=max(0, int(query.get("window_index", [0])[0])),
                    context_size=context_size,
                    stride=max(1, int(query.get("stride", [context_size])[0])),
                    patch_size=max(1, int(query.get("patch_size", [16])[0])),
                ),
            )
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_train_metrics(self, raw_query: str) -> None:
        try:
            query = parse_qs(raw_query)
            self._write_json(
                HTTPStatus.OK,
                build_train_metrics_payload(
                    output_dir_raw=str(query.get("output_dir", [""])[0]).strip() or None,
                    run_root_raw=str(query.get("run_root", [""])[0]).strip() or None,
                ),
            )
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _start_job(
        self,
        kind: str,
        payload: dict[str, Any],
        runner,
    ) -> None:
        """Create one background job, launch it, and return its id."""

        job = JOB_STORE.create_job(kind)
        thread = threading.Thread(
            target=JOB_STORE.run_job,
            args=(job, partial(runner, job_store=JOB_STORE), payload),
            daemon=True,
        )
        thread.start()
        self._write_json(HTTPStatus.ACCEPTED, {"job_id": job.job_id})

    def _read_json_body(self, optional: bool) -> Any:
        """Read and decode the request JSON body."""

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            if optional:
                return {}
            raise ValueError("Missing JSON request body")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, status: HTTPStatus, payload: Any) -> None:
        """Write one JSON response payload."""

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """Create the workbench HTTP server."""

    return ThreadingHTTPServer((host, port), WorkbenchRequestHandler)


def main() -> None:
    """Run the TSAD workbench server."""

    parser = argparse.ArgumentParser(description="TSAD Workbench interactive frontend")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8777, help="Port to bind")
    parser.add_argument(
        "--open-browser", action="store_true", help="Open the browser after startup"
    )
    args = parser.parse_args()

    server = make_server(args.host, int(args.port))
    url = f"http://{args.host}:{args.port}"
    print(f"TSAD Workbench listening on {url}")
    if args.open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

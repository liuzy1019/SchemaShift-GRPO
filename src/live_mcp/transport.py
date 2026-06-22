"""Subprocess stdio transport for local Live MCP servers."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from src.live_mcp import errors


class TransportError(RuntimeError):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


class MCPTransport(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def request(
        self,
        method: str,
        params: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]: ...


class SubprocessStdioTransport:
    """Line-delimited JSON RPC over a local subprocess stdio pair."""

    def __init__(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        startup_timeout_s: float = 20.0,
    ):
        self.argv = argv
        self.cwd = cwd
        self.env = env or {}
        self.startup_timeout_s = startup_timeout_s
        self.process: subprocess.Popen[str] | None = None
        self._responses: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._response_lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    @property
    def stderr_text(self) -> str:
        return "\n".join(self._stderr_lines[-50:])

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        child_env = os.environ.copy()
        child_env.update(self.env)
        self.process = subprocess.Popen(
            self.argv,
            cwd=str(self.cwd),
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader_thread.start()
        self._stderr_thread.start()
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            try:
                resp = self.request("healthcheck", {}, timeout_s=0.5)
            except TransportError:
                if self.process.poll() is not None:
                    raise
                continue
            if resp.get("ok"):
                return
        raise TransportError(errors.TIMEOUT, "server startup timed out")

    def stop(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            try:
                self.request("shutdown", {}, timeout_s=0.5)
            except TransportError:
                pass
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        self.process = None

    def request(
        self,
        method: str,
        params: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        if not self.process or self.process.poll() is not None:
            raise TransportError(errors.SERVER_UNAVAILABLE, self.stderr_text or "server not running")
        if not self.process.stdin:
            raise TransportError(errors.SERVER_UNAVAILABLE, "server stdin unavailable")
        req_id = uuid.uuid4().hex
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._response_lock:
            self._responses[req_id] = q
        request = {"id": req_id, "method": method, "params": params}
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
            self.process.stdin.flush()
            response = q.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TransportError(errors.TIMEOUT, f"request timed out: {method}") from exc
        except (BrokenPipeError, OSError) as exc:
            raise TransportError(errors.SERVER_UNAVAILABLE, str(exc)) from exc
        finally:
            with self._response_lock:
                self._responses.pop(req_id, None)
        if "error" in response:
            error = response.get("error") or {}
            raise TransportError(
                str(error.get("type") or errors.EXECUTION_ERROR),
                str(error.get("message") or "request failed"),
            )
        return response.get("result", {})

    def _read_stdout(self) -> None:
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = response.get("id")
            if not req_id:
                continue
            with self._response_lock:
                q = self._responses.get(req_id)
            if q is not None:
                q.put(response)

    def _read_stderr(self) -> None:
        if not self.process or not self.process.stderr:
            return
        for line in self.process.stderr:
            self._stderr_lines.append(line.rstrip())


class InProcessTransport:
    """Test/debug transport backed by an object with handle_request()."""

    def __init__(self, handler: Any):
        self.handler = handler
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def request(
        self,
        method: str,
        params: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        if not self.started:
            raise TransportError(errors.SERVER_UNAVAILABLE, "in-process transport not started")
        response = self.handler.handle_request(method, params)
        if "error" in response:
            error = response["error"]
            raise TransportError(error.get("type", errors.EXECUTION_ERROR), error.get("message", ""))
        return response.get("result", {})

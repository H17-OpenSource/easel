"""A small client for ComfyUI's HTTP API, written on the standard library (urllib) only.

Why a client of our own: the tools talk to ComfyUI over HTTP and nothing else (it is a separate GPL-3.0 program;
it is never imported, vendored or copied), and the few calls they need fit in one file with no dependency.

Where the server is: the base URL comes from the instance configuration, key `comfyui.url`
(`Client.from_config(cfg)`); without one it is ComfyUI's own upstream default, `DEFAULT_URL`. Nothing here knows
a host port of its own: the tool's default port is not necessarily the server's, so set `comfyui.url` when
the server listens elsewhere.

The calls (ComfyUI's public endpoints):
- `system_stats()`            GET /system_stats       version, Python, devices; `ping()` is the short-timeout form
- `object_info(cls=None)`     GET /object_info[/cls]  node classes and their input names (the graph checks use it)
- `queue(graph) -> prompt_id` POST /prompt            queue an API-format graph
- `history(prompt_id)`        GET /history/<id>       the prompt's entry, or None while it is queued or running
- `wait(prompt_id, timeout)`  polls history           the finished entry; raises on failure or timeout
- `view(filename, subfolder, type) -> bytes`  GET /view  one output, input or temp file, byte for byte
- `upload_image(data, name)`  POST /upload/image      put an init image into the server's input folder
- `queue_state()`             GET /queue              how many prompts run and wait
- `interrupt(prompt_id=None)` POST /interrupt         stop the running prompt

Every failure is a `ComfyError` with `lines`, ready to print: what happened, then a `fix:` line. The subclasses
say which kind: `ComfyUnreachable` (nothing answered: refused, unknown host, no answer in time), `ComfyRejected`
(the server refused the graph; one line per failing node), `ComfyFailed` (the prompt ran and failed; `status`
is the run status to record: failed, oom or cancelled), `ComfyTimeout` (`wait` gave up; the prompt may still
run), `ComfyHTTPError` (any other HTTP status). The text of these lines is ours; the server's own messages are
quoted after a colon and shortened.

Example (the demo instance; nothing is rendered without a running server):
    client = Client.from_config(config.load(tool))      # tool: the running tool's name
    prompt_id = client.queue(graph)
    entry = client.wait(prompt_id, timeout=600)
    for image in output_images(entry):
        data = client.view(image["filename"], image["subfolder"], image["type"])
"""

from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable, Mapping

DEFAULT_URL = "http://127.0.0.1:8188"  # ComfyUI's upstream default; an instance sets comfyui.url
DEFAULT_TIMEOUT_S = 30.0
PING_TIMEOUT_S = 3.0
FILE_TIMEOUT_S = 120.0  # /view and /upload move whole images
POLL_S = 1.0
FILE_TYPES = ("output", "input", "temp")
_QUOTE_MAX = 300  # the longest piece of a server message quoted in an error line


# --- errors -----------------------------------------------------------------------------------------------------

class ComfyError(Exception):
    """A ComfyUI call failed. `lines` are ready to print (what happened, then a `fix:` line); `url` is the server."""

    def __init__(self, lines: str | Iterable[str], *, url: str = "") -> None:
        self.lines = [lines] if isinstance(lines, str) else list(lines)
        self.url = url
        super().__init__("\n".join(self.lines))


class ComfyUnreachable(ComfyError):
    """Nothing answered at the URL: the connection was refused, the host is unknown, or no answer came in time."""


class ComfyHTTPError(ComfyError):
    """The server answered with an HTTP error status. `status` is the code, `body` the decoded answer."""

    def __init__(self, lines: str | Iterable[str], *, url: str = "", status: int = 0, body: object = None) -> None:
        super().__init__(lines, url=url)
        self.status = status
        self.body = body


class ComfyRejected(ComfyError):
    """The server refused the graph. `node_errors` is the server's map; `prompt_id` is set when the valid rest of
    the graph was queued anyway (ComfyUI queues the outputs that passed validation)."""

    def __init__(self, lines: str | Iterable[str], *, url: str = "", node_errors: Mapping | None = None,
                 prompt_id: str | None = None) -> None:
        super().__init__(lines, url=url)
        self.node_errors = dict(node_errors or {})
        self.prompt_id = prompt_id


class ComfyFailed(ComfyError):
    """The prompt ran and failed. `status` is the run status to record: `failed`, `oom` or `cancelled`."""

    def __init__(self, lines: str | Iterable[str], *, url: str = "", prompt_id: str = "", status: str = "failed",
                 entry: Mapping | None = None) -> None:
        super().__init__(lines, url=url)
        self.prompt_id = prompt_id
        self.status = status
        self.entry = dict(entry or {})


class ComfyTimeout(ComfyError):
    """`wait` gave up before the prompt finished; the prompt may still be queued or running on the server."""

    def __init__(self, lines: str | Iterable[str], *, url: str = "", prompt_id: str = "") -> None:
        super().__init__(lines, url=url)
        self.prompt_id = prompt_id


# --- small helpers ------------------------------------------------------------------------------------------------

def _short(text: object, limit: int = _QUOTE_MAX) -> str:
    """One line of at most `limit` characters, for quoting a server message inside our own line."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def normalise_url(url: object) -> str:
    """The base URL without a trailing slash; anything but http(s)://host[:port][/prefix] is a ComfyError."""
    text = str(url or "").strip().rstrip("/")
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.query or parts.fragment:
        raise ComfyError([f"comfyui.url must be an http(s) address such as {DEFAULT_URL}, not {url!r}",
                          "fix: set it with `config set comfyui.url <url>`"])
    return text


def _is_loopback(host: str) -> bool:
    return host in ("localhost", "::1") or host.startswith("127.")


def entry_state(entry: Mapping | None) -> str:
    """The state of a history entry: `pending` (no entry yet, or not finished), `success` or `error`."""
    if not entry:
        return "pending"
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        return "error"
    if status.get("completed") is True:
        return "success"
    if "completed" not in status and status.get("status_str") == "success":
        return "success"
    if not status and entry.get("outputs"):  # an older server that writes no status block
        return "success"
    return "pending"


def _messages(entry: Mapping) -> list[tuple[str, dict]]:
    """The entry's status messages as (kind, data) pairs; malformed items are skipped."""
    out = []
    for item in (entry.get("status") or {}).get("messages") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], Mapping):
            out.append((str(item[0]), dict(item[1])))
    return out


def failure(entry: Mapping, prompt_id: str) -> tuple[str, list[str]]:
    """(run status, lines) for a failed history entry: `cancelled` when interrupted, `oom` when the error is an
    out-of-memory error, otherwise `failed`."""
    messages = _messages(entry)
    error = next((data for kind, data in messages if kind == "execution_error"), None)
    if any(kind == "execution_interrupted" for kind, _ in messages) and error is None:
        return "cancelled", [f"{prompt_id}: the prompt was interrupted before it finished",
                             "fix: submit it again when the server is free"]
    if error is None:
        return "failed", [f"{prompt_id}: ComfyUI reported an error without details",
                          "fix: read the server's log, then submit again once"]
    kind = str(error.get("exception_type") or "error")
    text = str(error.get("exception_message") or "").strip()
    oom = "out of memory" in text.lower() or kind.lower().endswith("outofmemoryerror")
    where = f"node {error.get('node_id', '?')} ({error.get('node_type', '?')})"
    lines = [f"{prompt_id}: ComfyUI reported an error in {where}: {kind}: {_short(text.splitlines()[0] if text else '')}"]
    if oom:
        lines.append("fix: the graph did not fit the GPU; render fewer images at once, a smaller size or without "
                     "the hires pass")
        return "oom", lines
    lines.append("fix: read the full message in the server's log; do not resubmit the same graph unchanged")
    return "failed", lines


def rejection_lines(answer: Mapping, graph: Mapping | None = None) -> list[str]:
    """Our lines for the server's refusal of a graph: its reason, then one line per error per node."""
    error = answer.get("error") if isinstance(answer.get("error"), Mapping) else {}
    reason = error.get("message") or error.get("type") or answer.get("error") or "no reason given"
    head = f"ComfyUI rejected the graph: {_short(reason)}"
    if error.get("details"):
        head += f" ({_short(error['details'])})"
    lines = [head]
    node_errors = answer.get("node_errors") if isinstance(answer.get("node_errors"), Mapping) else {}
    for node_id, info in node_errors.items():
        info = info if isinstance(info, Mapping) else {}
        cls = info.get("class_type")
        if not cls and graph and isinstance(graph.get(node_id), Mapping):
            cls = graph[node_id].get("class_type")
        for err in info.get("errors") or [{}]:
            err = err if isinstance(err, Mapping) else {"message": str(err)}
            line = f"  node {node_id} ({cls or '?'}): {_short(err.get('message') or err.get('type') or 'invalid')}"
            if err.get("details"):
                line += f": {_short(err['details'])}"
            lines.append(line)
    lines.append("fix: report it to the operator: a missing model name means the file is not where the server "
                 "looks; a missing node class means its node pack is not installed")
    return lines


def output_images(entry: Mapping | None, type: str | None = "output") -> list[dict]:
    """The images a finished prompt saved, in the server's order: [{node, filename, subfolder, type}].
    `type` filters (`output` keeps saved files, dropping previews); None keeps every image."""
    found = []
    for node_id, out in ((entry or {}).get("outputs") or {}).items():
        for image in (out or {}).get("images") or []:
            if type is not None and image.get("type") != type:
                continue
            found.append({"node": str(node_id), "filename": image.get("filename", ""),
                          "subfolder": image.get("subfolder", ""), "type": image.get("type", "")})
    return found


def _multipart(fields: Mapping[str, str], file_field: str, file_name: str, data: bytes) -> tuple[bytes, str]:
    """A multipart/form-data body and its content type; the file's bytes are sent as they are."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{file_name}"\r\n'
                 "Content-Type: application/octet-stream\r\n\r\n".encode())
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# --- the client ---------------------------------------------------------------------------------------------------

class Client:
    """One ComfyUI server at `url`. `tool` ("easel" or "kiln") only shapes the fix lines of errors."""

    def __init__(self, url: str = DEFAULT_URL, *, timeout: float = DEFAULT_TIMEOUT_S, client_id: str | None = None,
                 tool: str | None = None) -> None:
        self.url = normalise_url(url)
        self.timeout = float(timeout)
        self.client_id = client_id or uuid.uuid4().hex
        self.tool = tool
        host = urllib.parse.urlsplit(self.url).hostname or ""
        # A server on this machine is reached directly, never through a proxy from the environment.
        handlers = [urllib.request.ProxyHandler({})] if _is_loopback(host) else []
        self._opener = urllib.request.build_opener(*handlers)

    @classmethod
    def from_config(cls, cfg: object, **kwargs: object) -> Client:
        """A client for the configured server: `comfyui.url`, or `DEFAULT_URL` when the key is not set."""
        url = cfg.get("comfyui.url", DEFAULT_URL)  # type: ignore[attr-defined]
        return cls(str(url or DEFAULT_URL), tool=getattr(cfg, "tool", None), **kwargs)  # type: ignore[arg-type]

    # -- plumbing --

    def url_for(self, path: str, query: Mapping[str, object] | None = None) -> str:
        """The full URL of an endpoint path, with its query string."""
        url = self.url + path
        if query:
            url += "?" + urllib.parse.urlencode({k: "" if v is None else str(v) for k, v in query.items()})
        return url

    def _fix_url_line(self) -> str:
        cmd = f"{self.tool} config set comfyui.url <url>" if self.tool else "config set comfyui.url <url>"
        return (f"fix: when the server runs at another address, point comfyui.url at it: `{cmd}` (ComfyUI's own "
                f"default is {DEFAULT_URL}); when it should be running here, report it to the operator")

    def _unreachable(self, reason: object, timeout: float) -> ComfyUnreachable:
        if isinstance(reason, TimeoutError):
            what = f"no answer within {timeout:g} s (the server may be busy loading a model)"
        elif isinstance(reason, ConnectionRefusedError):
            what = "the connection was refused (nothing listens there)"
        elif isinstance(reason, socket.gaierror):
            what = "the host name does not resolve"
        else:
            what = f"{type(reason).__name__}: {_short(reason)}"
        return ComfyUnreachable([f"ComfyUI is not reachable at {self.url}: {what}", self._fix_url_line()], url=self.url)

    def _http_error(self, method: str, path: str, exc: urllib.error.HTTPError) -> ComfyHTTPError:
        try:
            raw = exc.read()
        except OSError:
            raw = b""
        text = raw.decode("utf-8", "replace")
        try:
            body: object = json.loads(text) if text else None
        except ValueError:
            body = text
        detail = f": {_short(text)}" if text else ""
        line = f"{method} {path}: the server answered HTTP {exc.code} {exc.reason}{detail}"
        return ComfyHTTPError([line, "fix: check the server's log and version (`system_stats`)"], url=self.url,
                              status=exc.code, body=body)

    def _request(self, method: str, path: str, *, query: Mapping[str, object] | None = None, body: object = None,
                 data: bytes | None = None, content_type: str | None = None, timeout: float | None = None,
                 want: str = "json") -> object:
        """One HTTP call. `body` is sent as JSON, `data` as raw bytes; the answer is JSON (`want="json"`) or bytes."""
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        limit = self.timeout if timeout is None else float(timeout)
        request = urllib.request.Request(self.url_for(path, query), data=data, method=method, headers=headers)
        try:
            with self._opener.open(request, timeout=limit) as answer:
                raw = answer.read()
        except urllib.error.HTTPError as exc:  # before URLError: HTTPError is one
            raise self._http_error(method, path, exc) from None
        except urllib.error.URLError as exc:
            raise self._unreachable(exc.reason, limit) from None
        except http.client.HTTPException as exc:
            raise ComfyError([f"{method} {path}: something answered at {self.url}, but not in HTTP "
                              f"({type(exc).__name__}); is it a ComfyUI server?", self._fix_url_line()],
                             url=self.url) from None
        except OSError as exc:  # TimeoutError, ConnectionResetError and the like, during the call or the read
            raise self._unreachable(exc, limit) from None
        if want == "bytes":
            return raw
        try:
            decoded = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except ValueError:
            decoded = None
        if not isinstance(decoded, dict):
            raise ComfyError([f"{method} {path}: the answer from {self.url} is not a JSON object; is it a ComfyUI "
                              "server?", self._fix_url_line()], url=self.url)
        return decoded

    # -- the calls --

    def system_stats(self, timeout: float | None = None) -> dict:
        """GET /system_stats: {"system": {"comfyui_version", "python_version", ...}, "devices": [...]}."""
        return dict(self._request("GET", "/system_stats", timeout=timeout))

    def ping(self) -> dict | None:
        """system_stats with a short timeout, or None when nothing answers (for status lines and doctor)."""
        try:
            return self.system_stats(timeout=PING_TIMEOUT_S)
        except ComfyUnreachable:
            return None

    def version(self) -> str | None:
        """The server's ComfyUI version, as it reports it (None when it does not say)."""
        return (self.system_stats().get("system") or {}).get("comfyui_version")

    def object_info(self, node_class: str | None = None) -> dict:
        """GET /object_info (every node class; large on a full install) or /object_info/<class> (one class; the
        answer is {} when the server does not know the class)."""
        path = "/object_info" if node_class is None else "/object_info/" + urllib.parse.quote(node_class, safe="")
        return dict(self._request("GET", path))

    def object_info_for(self, classes: Iterable[str]) -> dict:
        """The node definitions of the given classes, one call per class, merged; unknown classes are absent."""
        merged: dict = {}
        for cls in dict.fromkeys(classes):
            merged.update(self.object_info(cls))
        return merged

    def queue(self, graph: Mapping, *, front: bool = False) -> str:
        """POST /prompt: queue an API-format graph and return its prompt id. A refused graph is ComfyRejected."""
        body: dict = {"prompt": dict(graph), "client_id": self.client_id}
        if front:
            body["front"] = True
        try:
            answer = self._request("POST", "/prompt", body=body)
        except ComfyHTTPError as exc:
            if exc.status == 400 and isinstance(exc.body, Mapping):
                raise ComfyRejected(rejection_lines(exc.body, graph), url=self.url,
                                    node_errors=exc.body.get("node_errors")) from None
            raise
        prompt_id = answer.get("prompt_id") if isinstance(answer, Mapping) else None
        if not prompt_id:
            raise ComfyError([f"POST /prompt: the answer from {self.url} carries no prompt_id",
                              "fix: check the server's version and log"], url=self.url)
        node_errors = answer.get("node_errors") or {}
        if node_errors:
            lines = rejection_lines(answer, graph)
            lines.insert(1, f"  the outputs that passed were queued as {prompt_id}")
            raise ComfyRejected(lines, url=self.url, node_errors=node_errors, prompt_id=str(prompt_id))
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict | None:
        """GET /history/<id>: the prompt's entry ({"outputs", "status", ...}), or None while it has none yet."""
        answer = self._request("GET", "/history/" + urllib.parse.quote(str(prompt_id), safe=""))
        entry = answer.get(str(prompt_id)) if isinstance(answer, Mapping) else None
        return dict(entry) if isinstance(entry, Mapping) else None

    def wait(self, prompt_id: str, timeout: float, *, poll: float = POLL_S) -> dict:
        """Poll the history until the prompt finishes and return its entry.

        Raises ComfyFailed (it ran and failed; `.status` says failed, oom or cancelled), ComfyTimeout (no result
        within `timeout` seconds; it may still run) or ComfyUnreachable (the server went away while waiting).
        """
        deadline = time.monotonic() + float(timeout)
        while True:
            try:
                entry = self.history(prompt_id)
            except ComfyUnreachable as exc:
                note = (f"while waiting for {prompt_id}: the server stopped answering (crashed or stopped?); "
                        "the prompt's result is unknown")
                raise ComfyUnreachable([exc.lines[0], note, *exc.lines[1:]], url=self.url) from None
            state = entry_state(entry)
            if state == "success":
                return entry or {}
            if state == "error":
                status, lines = failure(entry or {}, prompt_id)
                raise ComfyFailed(lines, url=self.url, prompt_id=prompt_id, status=status, entry=entry)
            left = deadline - time.monotonic()
            if left <= 0:
                raise ComfyTimeout([f"{prompt_id}: no result after {float(timeout):g} s",
                                    "fix: it may still be queued or running; look at the server's queue "
                                    "(`queue_state`) before submitting again"], url=self.url, prompt_id=prompt_id)
            time.sleep(min(float(poll), left))

    def view(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:
        """GET /view: one file the server holds (`type` output, input or temp), byte for byte."""
        if type not in FILE_TYPES:
            raise ComfyError([f"view: type must be one of {', '.join(FILE_TYPES)}, not {type!r}"], url=self.url)
        try:
            return self._request("GET", "/view", query={"filename": filename, "subfolder": subfolder, "type": type},
                                 timeout=max(self.timeout, FILE_TIMEOUT_S), want="bytes")
        except ComfyHTTPError as exc:
            if exc.status == 404:
                where = f"{subfolder}/{filename}" if subfolder else filename
                raise ComfyHTTPError([f"the server has no {type} file {where}",
                                      "fix: take the names from the prompt's history (output_images)"],
                                     url=self.url, status=404, body=exc.body) from None
            raise

    def upload_image(self, data: bytes, name: str, *, subfolder: str = "", overwrite: bool = True) -> str:
        """POST /upload/image: put an image into the server's input folder; returns the name LoadImage takes.
        The bytes are sent as they are; nothing here decodes or looks at the image."""
        fields = {"overwrite": "true" if overwrite else "false", "type": "input"}
        if subfolder:
            fields["subfolder"] = subfolder
        payload, ctype = _multipart(fields, "image", name, data)
        answer = self._request("POST", "/upload/image", data=payload, content_type=ctype,
                               timeout=max(self.timeout, FILE_TIMEOUT_S))
        stored = answer.get("name") if isinstance(answer, Mapping) else None
        if not stored:
            raise ComfyError([f"POST /upload/image: the answer from {self.url} names no stored file"], url=self.url)
        folder = answer.get("subfolder") or ""
        return f"{folder}/{stored}" if folder else str(stored)

    def queue_state(self) -> dict:
        """GET /queue as counts: {"running": n, "pending": n}."""
        answer = self._request("GET", "/queue")
        return {"running": len(answer.get("queue_running") or []), "pending": len(answer.get("queue_pending") or [])}

    def interrupt(self, prompt_id: str | None = None) -> None:
        """POST /interrupt: stop the running prompt (newer servers stop only `prompt_id` when it is given)."""
        self._request("POST", "/interrupt", body={"prompt_id": prompt_id} if prompt_id else {})

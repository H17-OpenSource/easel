"""_common.comfy_client against the conftest `mock_comfy` server (and a few raw sockets for the failure cases).

No test reaches a real ComfyUI: the mock answers on 127.0.0.1 and a free port, the `no_network` fixture refuses
anything else, and every port below is picked by the OS at run time.
"""

import socket
import threading
import urllib.parse

import pytest

from easel._common import comfy_client as cc
from easel._common import config

PROMPT_ID = "00000000-0000-4000-8000-000000000001"  # the mock's answer to POST /prompt


class StubConfig:
    """The two things Client.from_config reads from a loaded configuration."""

    def __init__(self, values: dict, tool: str = "easel") -> None:
        self.values, self.tool = values, tool

    def get(self, key, default=None):
        return self.values.get(key, default)


def free_port() -> int:
    """A port nothing listens on: bound by the OS, then released."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def client(mock_comfy, no_network):
    return cc.Client(mock_comfy.url, tool="easel", timeout=5)


@pytest.fixture
def silent_url():
    """A port that accepts connections and never answers (the kernel completes the handshake)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    try:
        yield f"http://127.0.0.1:{s.getsockname()[1]}"
    finally:
        s.close()


def entry(status_str="success", completed=True, messages=(), outputs=None):
    return {"prompt": [], "outputs": outputs or {}, "status": {"status_str": status_str, "completed": completed,
                                                               "messages": [list(m) for m in messages]}}


# --- where the server is ------------------------------------------------------------------------------------------

def test_default_url_is_comfyuis_upstream_default():
    assert cc.Client().url == cc.DEFAULT_URL
    assert cc.DEFAULT_URL.startswith("http://127.0.0.1:")


def test_from_config_reads_comfyui_url_and_falls_back_to_the_default(mock_comfy):
    assert cc.Client.from_config(StubConfig({"comfyui.url": mock_comfy.url + "/"})).url == mock_comfy.url
    stub = StubConfig({}, tool="kiln")
    made = cc.Client.from_config(stub)
    assert made.url == cc.DEFAULT_URL and made.tool == "kiln"


def test_from_config_on_the_demo_instance(demo_workspace):
    cfg = config.load("easel", demo_workspace, strict=False)
    made = cc.Client.from_config(cfg)
    assert made.url == cc.DEFAULT_URL
    assert made.tool == "easel"


@pytest.mark.parametrize("bad", ["", "ftp://127.0.0.1", "not a url", cc.DEFAULT_URL.split("//", 1)[1],
                                 cc.DEFAULT_URL + "/?x=1"])
def test_a_url_that_is_not_http_is_refused_with_the_fix(bad):
    with pytest.raises(cc.ComfyError) as err:
        cc.Client(bad)
    assert "comfyui.url must be an http(s) address" in err.value.lines[0]
    assert err.value.lines[-1].startswith("fix:")


def test_url_for_builds_the_query_string(client):
    query = {"filename": "demo 001.png", "subfolder": "easel/run-20300101-001", "type": "output"}
    url = client.url_for("/view", query)
    parts = urllib.parse.urlsplit(url)
    assert url.startswith(client.url + "/view?") and " " not in url
    assert urllib.parse.parse_qs(parts.query) == {k: [v] for k, v in query.items()}
    assert list(urllib.parse.parse_qs(parts.query)) == ["filename", "subfolder", "type"]
    assert client.url_for("/queue") == client.url + "/queue"


# --- an unreachable server --------------------------------------------------------------------------------------------

def test_a_refused_connection_is_a_clear_unreachable_error(no_network):
    url = f"http://127.0.0.1:{free_port()}"
    with pytest.raises(cc.ComfyUnreachable) as err:
        cc.Client(url, tool="easel", timeout=10).system_stats()
    lines = err.value.lines
    assert lines[0].startswith(f"ComfyUI is not reachable at {url}: ")
    assert "refused" in lines[0] or "no answer" in lines[0]
    assert lines[1].startswith("fix: ") and "easel config set comfyui.url <url>" in lines[1]
    assert err.value.url == url


def test_ping_is_none_when_nothing_answers(no_network):
    assert cc.Client(f"http://127.0.0.1:{free_port()}").ping() is None


def test_no_answer_in_time_is_unreachable(silent_url, no_network):
    with pytest.raises(cc.ComfyUnreachable) as err:
        cc.Client(silent_url, timeout=0.3).system_stats()
    assert "no answer within 0.3 s" in err.value.lines[0]


def test_something_that_is_not_http_is_named(no_network):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def answer_garbage():
        conn, _ = server.accept()
        with conn:
            conn.recv(4096)
            conn.sendall(b"hello, this is not HTTP\r\n\r\n")

    thread = threading.Thread(target=answer_garbage, daemon=True)
    thread.start()
    try:
        with pytest.raises(cc.ComfyError) as err:
            cc.Client(f"http://127.0.0.1:{server.getsockname()[1]}", timeout=5).system_stats()
    finally:
        thread.join(timeout=5)
        server.close()
    assert "is it a ComfyUI server?" in str(err.value)


def test_an_answer_that_is_not_json_is_named(client, mock_comfy):
    mock_comfy.routes[("GET", "/system_stats")] = (200, b"<html>a web page</html>")
    with pytest.raises(cc.ComfyError) as err:
        client.system_stats()
    assert "is not a JSON object; is it a ComfyUI server?" in err.value.lines[0]


def test_env_proxies_are_bypassed_for_a_local_server(client, monkeypatch):
    dead_proxy = f"http://127.0.0.1:{free_port()}"
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, dead_proxy)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    fresh = cc.Client(client.url, timeout=5)
    assert fresh.version() == "0.43.1"


# --- the calls ------------------------------------------------------------------------------------------------------

def test_system_stats_ping_and_version(client, mock_comfy):
    stats = client.system_stats()
    assert stats["system"]["comfyui_version"] == "0.43.1"
    assert client.ping() == stats
    assert client.version() == "0.43.1"
    assert [r[:2] for r in mock_comfy.requests] == [("GET", "/system_stats")] * 3


def test_object_info_all_one_class_and_many(client, mock_comfy):
    mock_comfy.routes[("GET", "/object_info")] = (200, {"KSampler": {"input": {}}, "SaveImage": {"input": {}}})
    mock_comfy.routes[("GET", "/object_info/KSampler")] = (200, {"KSampler": {"input": {"required": {"seed": ["*"]}}}})
    mock_comfy.routes[("GET", "/object_info/SaveImage")] = (200, {"SaveImage": {"input": {}}})
    assert set(client.object_info()) == {"KSampler", "SaveImage"}
    assert client.object_info("KSampler")["KSampler"]["input"]["required"] == {"seed": ["*"]}
    merged = client.object_info_for(["KSampler", "SaveImage", "KSampler", "NoSuchNode"])
    assert set(merged) == {"KSampler", "SaveImage"}  # an unknown class answers {} and is absent
    paths = [p for _, p, _ in mock_comfy.requests]
    assert paths == ["/object_info", "/object_info/KSampler", "/object_info/KSampler", "/object_info/SaveImage",
                     "/object_info/NoSuchNode"]  # one call per distinct class


def test_object_info_quotes_a_class_name_with_a_space(client, mock_comfy):
    client.object_info("Demo Node")
    path = mock_comfy.requests[-1][1]
    assert " " not in path and urllib.parse.unquote(path) == "/object_info/Demo Node"


def test_queue_posts_the_graph_with_the_client_id_and_returns_the_prompt_id(client, mock_comfy):
    graph = {"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "demo"}}}
    assert client.queue(graph) == PROMPT_ID
    method, path, body = mock_comfy.requests[-1]
    assert (method, path) == ("POST", "/prompt")
    assert body == {"prompt": graph, "client_id": client.client_id}
    client.queue(graph, front=True)
    assert mock_comfy.requests[-1][2]["front"] is True


def test_a_rejected_graph_lists_every_node_error(client, mock_comfy):
    refusal = {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                         "details": "", "extra_info": {}},
               "node_errors": {"1": {"errors": [{"type": "value_not_in_list", "message": "Value not in list",
                                                 "details": "ckpt_name: 'missing.safetensors' not in []"}],
                                     "dependent_outputs": ["9"], "class_type": "CheckpointLoaderSimple"},
                               "6": {"errors": [{"message": "Required input is missing"}]}}}
    mock_comfy.routes[("POST", "/prompt")] = (400, refusal)
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}}, "6": {"class_type": "KSampler", "inputs": {}}}
    with pytest.raises(cc.ComfyRejected) as err:
        client.queue(graph)
    lines = err.value.lines
    assert lines[0] == "ComfyUI rejected the graph: Prompt outputs failed validation"
    assert lines[1] == "  node 1 (CheckpointLoaderSimple): Value not in list: ckpt_name: 'missing.safetensors' not in []"
    assert lines[2] == "  node 6 (KSampler): Required input is missing"  # the class comes from our graph
    assert lines[-1].startswith("fix: ")
    assert set(err.value.node_errors) == {"1", "6"} and err.value.prompt_id is None


def test_a_partly_valid_graph_is_rejected_with_the_queued_prompt_id(client, mock_comfy):
    answer = {"prompt_id": PROMPT_ID, "number": 2,
              "node_errors": {"4": {"errors": [{"message": "Value not in list"}], "class_type": "VAELoader"}}}
    mock_comfy.routes[("POST", "/prompt")] = (200, answer)
    with pytest.raises(cc.ComfyRejected) as err:
        client.queue({"4": {"class_type": "VAELoader", "inputs": {}}})
    assert err.value.prompt_id == PROMPT_ID
    assert f"queued as {PROMPT_ID}" in err.value.lines[1]


def test_another_http_error_keeps_status_and_body(client, mock_comfy):
    mock_comfy.routes[("POST", "/prompt")] = (500, {"error": "boom"})
    with pytest.raises(cc.ComfyHTTPError) as err:
        client.queue({})
    assert err.value.status == 500 and err.value.body == {"error": "boom"}
    assert "HTTP 500" in err.value.lines[0]


def test_history_is_the_entry_or_none(client, mock_comfy):
    assert client.history(PROMPT_ID) is None  # the mock's default: {} while queued
    mock_comfy.routes[("GET", "/history/")] = (200, {PROMPT_ID: entry()})
    assert client.history(PROMPT_ID)["status"]["completed"] is True
    assert mock_comfy.requests[-1][1] == f"/history/{PROMPT_ID}"


@pytest.mark.parametrize(("item", "state"), [
    (None, "pending"),
    ({}, "pending"),
    (entry("success", False), "pending"),
    (entry("success", True), "success"),
    ({"status": {"status_str": "success"}}, "success"),  # no completed flag: the status word decides
    (entry("error", True), "error"),
    ({"outputs": {"9": {"images": []}}}, "success"),  # an older server without a status block
])
def test_entry_state(item, state):
    assert cc.entry_state(item) == state


def test_wait_returns_the_finished_entry(client, mock_comfy):
    mock_comfy.routes[("GET", "/history/")] = (200, {PROMPT_ID: entry()})
    assert client.wait(PROMPT_ID, timeout=5, poll=0.01)["status"]["status_str"] == "success"


def test_wait_polls_until_the_prompt_finishes(client, mock_comfy):
    timer = threading.Timer(0.3, lambda: mock_comfy.routes.__setitem__(("GET", "/history/"), (200, {PROMPT_ID: entry()})))
    timer.start()
    try:
        result = client.wait(PROMPT_ID, timeout=10, poll=0.05)
    finally:
        timer.cancel()
    assert result["status"]["completed"] is True
    assert sum(1 for _, p, _ in mock_comfy.requests if p.startswith("/history/")) >= 2


def test_wait_times_out_with_the_prompt_id(client):
    with pytest.raises(cc.ComfyTimeout) as err:
        client.wait(PROMPT_ID, timeout=0.2, poll=0.05)
    assert err.value.prompt_id == PROMPT_ID
    assert err.value.lines[0] == f"{PROMPT_ID}: no result after 0.2 s"
    assert "may still be queued or running" in err.value.lines[1]


@pytest.mark.parametrize(("message", "status"), [
    ("Allocation on device: CUDA out of memory (synthetic)", "oom"),
    ("synthetic failure", "failed"),
])
def test_wait_raises_the_failure_with_the_run_status(client, mock_comfy, message, status):
    error = {"prompt_id": PROMPT_ID, "node_id": "7", "node_type": "KSampler", "exception_type": "RuntimeError",
             "exception_message": message + "\nsecond line"}
    mock_comfy.routes[("GET", "/history/")] = (200, {PROMPT_ID: entry("error", True, [("execution_start", {}),
                                                                                       ("execution_error", error)])})
    with pytest.raises(cc.ComfyFailed) as err:
        client.wait(PROMPT_ID, timeout=5, poll=0.01)
    assert err.value.status == status and err.value.prompt_id == PROMPT_ID
    assert err.value.lines[0] == f"{PROMPT_ID}: ComfyUI reported an error in node 7 (KSampler): RuntimeError: {message}"
    assert err.value.lines[-1].startswith("fix: ")


def test_an_interrupted_prompt_is_cancelled(client, mock_comfy):
    mock_comfy.routes[("GET", "/history/")] = (200, {PROMPT_ID: entry("error", True, [("execution_interrupted",
                                                                                       {"node_id": "7"})])})
    with pytest.raises(cc.ComfyFailed) as err:
        client.wait(PROMPT_ID, timeout=5, poll=0.01)
    assert err.value.status == "cancelled"


def test_wait_says_when_the_server_went_away(silent_url, no_network):
    with pytest.raises(cc.ComfyUnreachable) as err:
        cc.Client(silent_url, timeout=0.3).wait(PROMPT_ID, timeout=5, poll=0.01)
    assert err.value.lines[0].startswith(f"ComfyUI is not reachable at {silent_url}")
    assert err.value.lines[1].startswith(f"while waiting for {PROMPT_ID}: the server stopped answering")
    assert err.value.lines[-1].startswith("fix: ")


def test_output_images_keeps_saved_files_in_the_servers_order():
    outputs = {"9": {"images": [{"filename": "demo_00001_.png", "subfolder": "easel", "type": "output"},
                                {"filename": "preview.png", "subfolder": "", "type": "temp"}]},
               "12": {"images": [{"filename": "demo_00002_.png", "subfolder": "easel", "type": "output"}]}}
    found = cc.output_images(entry(outputs=outputs))
    assert [(i["node"], i["filename"]) for i in found] == [("9", "demo_00001_.png"), ("12", "demo_00002_.png")]
    assert len(cc.output_images(entry(outputs=outputs), type=None)) == 3
    assert cc.output_images(None) == []


def test_view_returns_the_bytes_unchanged(client, mock_comfy):
    data = bytes(range(256)) * 4
    mock_comfy.routes[("GET", "/view")] = (200, data)
    assert client.view("demo_00001_.png", "easel", "output") == data
    assert mock_comfy.requests[-1][:2] == ("GET", "/view")


def test_view_of_a_missing_file_and_a_bad_type(client, mock_comfy):
    mock_comfy.routes[("GET", "/view")] = (404, {"error": "not found"})
    with pytest.raises(cc.ComfyHTTPError) as err:
        client.view("gone.png", "easel")
    assert err.value.status == 404 and err.value.lines[0] == "the server has no output file easel/gone.png"
    with pytest.raises(cc.ComfyError):
        client.view("demo.png", type="secret")


def test_upload_image_sends_multipart_and_returns_the_loadimage_name(client, mock_comfy):
    data = b"\x89PNG demo bytes"
    mock_comfy.routes[("POST", "/upload/image")] = (200, {"name": "init.png", "subfolder": "", "type": "input"})
    assert client.upload_image(data, "init.png") == "init.png"
    method, path, body = mock_comfy.requests[-1]
    assert (method, path) == ("POST", "/upload/image")
    assert isinstance(body, bytes) and data in body and b'filename="init.png"' in body and b'name="overwrite"' in body
    mock_comfy.routes[("POST", "/upload/image")] = (200, {"name": "init.png", "subfolder": "demo", "type": "input"})
    assert client.upload_image(data, "init.png", subfolder="demo") == "demo/init.png"


def test_queue_state_counts(client, mock_comfy):
    assert client.queue_state() == {"running": 0, "pending": 0}
    mock_comfy.routes[("GET", "/queue")] = (200, {"queue_running": [[1, "a"]], "queue_pending": [[2, "b"], [3, "c"]]})
    assert client.queue_state() == {"running": 1, "pending": 2}


def test_interrupt_posts_an_optional_prompt_id(client, mock_comfy):
    client.interrupt()
    assert mock_comfy.requests[-1] == ("POST", "/interrupt", {})
    client.interrupt(PROMPT_ID)
    assert mock_comfy.requests[-1] == ("POST", "/interrupt", {"prompt_id": PROMPT_ID})

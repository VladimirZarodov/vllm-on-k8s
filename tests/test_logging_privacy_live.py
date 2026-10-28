#!/usr/bin/env python3
"""Live end-to-end checks against the vLLM deployment in Minikube.

Verifies:
  1. /health answers without a token (so Kubernetes probes work);
  2. API endpoints reject missing/incorrect Bearer tokens (401);
  3. an authorized /v1/chat/completions request succeeds;
  4. the unique prompt marker sent above never appears in cluster logs;
  5. the Bearer token itself never appears in cluster logs;
  6. the request fact (method + path + status) IS visible in the logs.

Prerequisites (see README): deployment is Ready; kubectl on PATH with the
current context pointed at Minikube; the environment variable
VLLM_TEST_API_KEY set to the value stored in the vllm-secrets Secret.

Usage (from the repo root, PowerShell):

    $env:VLLM_TEST_API_KEY = Get-Content .checks\vllm-api-key.txt
    python tests/test_logging_privacy_live.py

Exit code 0 means every check passed.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

NAMESPACE = "vllm"
SERVICE = "svc/vllm"
LOCAL_PORT = 18000
BASE = f"http://127.0.0.1:{LOCAL_PORT}"
MODEL = "qwen2.5-0.5b-instruct"
# Set --expect-received-request when the server runs with
# --enable-log-requests: then the sanitized "Received request" line is
# additionally required in the logs.
EXPECT_RECEIVED_REQUEST = "--expect-received-request" in sys.argv


def kubectl(*args: str, timeout: int = 90) -> str:
    proc = subprocess.run(
        ["kubectl", *args], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        sys.exit(f"kubectl {' '.join(args)} failed:\n{proc.stderr}")
    return proc.stdout


def http(method: str, path: str, token: str | None = None, body: dict | None = None):
    request = urllib.request.Request(BASE + path, method=method)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request.add_header("Content-Type", "application/json")
    try:
        # 900 s: first real generation on a 4-vCPU constrained Minikube VM can
        # take minutes; 300 s was too tight and produced a spurious timeout.
        with urllib.request.urlopen(request, data=data, timeout=900) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except OSError as exc:
        return None, str(exc)


def main() -> int:
    api_key = os.environ.get("VLLM_TEST_API_KEY", "").strip()
    if not api_key:
        sys.exit("VLLM_TEST_API_KEY is not set; export it first (see README).")
    marker = f"UNSEEN-PROMPT-{os.urandom(8).hex().upper()}-END"
    print(f"using prompt marker: {marker}")

    forwarder = subprocess.Popen(
        ["kubectl", "port-forward", "-n", NAMESPACE, SERVICE, f"{LOCAL_PORT}:8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))

    try:
        for _ in range(60):
            try:
                with socket.create_connection(("127.0.0.1", LOCAL_PORT), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            sys.exit("port-forward did not become ready")

        since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        status, _ = http("GET", "/health")
        check("GET /health without token -> 200 (K8s probes work)", status == 200, f"status={status}")

        status, _ = http("GET", "/v1/models")
        check("GET /v1/models without token -> 401", status == 401, f"status={status}")

        status, _ = http("GET", "/v1/models", token="totally-wrong-key")
        check("GET /v1/models with wrong token -> 401", status == 401, f"status={status}")

        status, body = http("GET", "/v1/models", token=api_key)
        check(
            "GET /v1/models with correct token -> 200 + model listed",
            status == 200 and MODEL in body,
            f"status={status}",
        )

        chat_body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": f"Remember this phrase: {marker}"},
                {"role": "user", "content": f"Repeat exactly the phrase: {marker}"},
            ],
            "max_tokens": 24,
            "temperature": 0.0,
        }

        status, body = http("POST", "/v1/chat/completions", body=dict(chat_body))
        check("POST /v1/chat/completions without token -> 401", status == 401, f"status={status}")

        status, body = http("POST", "/v1/chat/completions", token=api_key, body=chat_body)
        ok = status == 200
        try:
            answer_len = len(json.loads(body)["choices"][0]["message"]["content"])
        except Exception:
            answer_len = -1
        check("POST /v1/chat/completions authorized -> 200 + answer", ok and answer_len >= 0,
              f"status={status}")
        if not ok:
            print(f"       response: {body[:300]}")

        time.sleep(3)
        logs = kubectl(
            "logs", "-n", NAMESPACE, "deploy/vllm",
            "--all-containers", "--prefix", f"--since-time={since}",
        )
        check("prompt marker ABSENT from cluster logs", marker not in logs)
        check("API key ABSENT from cluster logs", api_key not in logs)
        check(
            "request fact visible in access log (method + path + status)",
            'POST /v1/chat/completions' in logs and "200" in logs,
        )
        check(
            "unauthorized attempts visible in access log",
            'POST /v1/chat/completions' in logs and "401" in logs,
        )
        if EXPECT_RECEIVED_REQUEST:
            check(
                "sanitized 'Received request' line present (enable-log-requests mode)",
                "Received request" in logs and "request accepted" in logs,
            )
    finally:
        forwarder.terminate()
        try:
            forwarder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            forwarder.kill()

    print("-" * 60)
    print("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

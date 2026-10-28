#!/usr/bin/env python3
"""Unit tests for the privacy log filter (vllm_privacy.privacy_logging).

Standard library only, no vLLM installation required.  Run from the repo
root either with pytest or directly:

    python tests/test_privacy_filter.py
"""

from __future__ import annotations

import json
import logging
import sys
import unittest
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm_privacy.privacy_logging import PrivacyFilter  # noqa: E402

MARKER = "ZZTOP-SECRET-42d9f1"

# Exact module logger name used by vLLM 0.32.0
# (vllm/entrypoints/serve/utils/request_logger.py).
REQUEST_LOGGER = "vllm.entrypoints.serve.utils.request_logger"
REQUEST_PATHNAME = "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/serve/utils/request_logger.py"


def emit(logger_name: str, message: str, *, pathname: str = "", args: tuple = ()) -> str:
    """Run one LogRecord through the PrivacyFilter and return captured output."""
    record = logging.LogRecord(
        name=logger_name,
        level=logging.INFO,
        pathname=pathname or f"/fake/{logger_name.replace('.', '/')}.py",
        lineno=1,
        msg=message,
        args=args,
        exc_info=None,
    )
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(PrivacyFilter())
    handler.handle(record)
    return stream.getvalue()


class PrivacyFilterTest(unittest.TestCase):
    # --- request_logger: what must be sanitized -------------------------------

    def test_received_request_is_kept_but_params_removed(self):
        out = emit(
            REQUEST_LOGGER,
            f"Received request chatcmpl-77: params: SamplingParams(stop=['{MARKER}']), "
            "lora_request: None.",
            pathname=REQUEST_PATHNAME,
        )
        self.assertIn("Received request chatcmpl-77", out)
        self.assertNotIn(MARKER, out)
        self.assertNotIn("SamplingParams", out)

    def test_prompt_debug_line_is_dropped(self):
        out = emit(
            REQUEST_LOGGER,
            f"Request chatcmpl-77 details: prompt: 'Say {MARKER}', "
            "prompt_token_ids: [1, 2, 3], prompt_embeds shape: None.",
            pathname=REQUEST_PATHNAME,
        )
        self.assertEqual(out, "")

    def test_response_line_is_dropped(self):
        out = emit(
            REQUEST_LOGGER,
            f"Generated response chatcmpl-77: output: '{MARKER} token...', "
            "finish_reason: stop",
            pathname=REQUEST_PATHNAME,
        )
        self.assertEqual(out, "")

    def test_json_body_line_is_dropped(self):
        out = emit(
            REQUEST_LOGGER,
            f'Request chatcmpl-77 JSON body: {{"messages": [{{"content": "{MARKER}"}}]}}',
            pathname=REQUEST_PATHNAME,
        )
        self.assertEqual(out, "")

    def test_lazy_args_message_is_dropped(self):
        out = emit(
            REQUEST_LOGGER,
            "Add request: prompt = %s",
            pathname=REQUEST_PATHNAME,
            args=(f"Please repeat {MARKER}",),
        )
        self.assertEqual(out, "")

    def test_authorization_header_is_never_logged(self):
        out = emit(
            REQUEST_LOGGER,
            "Request headers: {'authorization': 'Bearer sk-verysecret', 'x-key': '%s'}" % MARKER,
            pathname=REQUEST_PATHNAME,
        )
        self.assertEqual(out, "")
        self.assertNotIn("sk-verysecret", out)

    # --- everything else must keep working -----------------------------------

    def test_unrelated_vllm_logger_passes_through(self):
        out = emit("vllm.model_executor.model_loader", "Loaded weights from /models/model")
        self.assertIn("Loaded weights", out)

    def test_access_log_keeps_method_path_status(self):
        out = emit(
            "uvicorn.access",
            '127.0.0.1:52133 - "POST /v1/chat/completions HTTP/1.1" 200 OK',
            pathname="/usr/local/lib/python3.12/site-packages/uvicorn/_proto.py",
        )
        self.assertIn("POST /v1/chat/completions", out)
        self.assertIn("200", out)

    def test_startup_log_with_model_path_is_kept(self):
        out = emit("vllm.entrypoints.openai.api_server", "Starting vLLM API server on port 8000")
        self.assertIn("Starting vLLM API server", out)


class LoggingConfigTest(unittest.TestCase):
    def test_config_json_wires_privacy_filter(self):
        config = json.loads((REPO_ROOT / "logging_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["version"], 1)
        self.assertEqual(
            config["filters"]["privacy"]["()"],
            "vllm_privacy.privacy_logging.PrivacyFilter",
        )
        self.assertIn("privacy", config["handlers"]["vllm"]["filters"])
        for logger in ("vllm", "uvicorn", "uvicorn.error", "uvicorn.access"):
            self.assertIn(logger, config["loggers"])
            self.assertEqual(config["loggers"][logger]["handlers"], ["vllm"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

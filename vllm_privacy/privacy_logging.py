import logging
import re


class PrivacyFilter(logging.Filter):
    """Hide prompts, request bodies and generated text in vLLM logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        logger_name = record.name.lower()
        filename = record.filename.lower()
        pathname = record.pathname.lower()

        is_request_logger = (
            "request_logger" in logger_name
            or "request_logger.py" in filename
            or "request_logger.py" in pathname
        )

        # Leave unrelated logs untouched.
        if not is_request_logger:
            return True

        message = record.getMessage()

        # Keep only the request ID and the fact that the request was accepted.
        if message.startswith("Received request "):
            match = re.match(
                r"^Received request\s+([^:\s]+)",
                message,
            )
            request_id = match.group(1) if match else "unknown"

            record.msg = f"Received request {request_id}: request accepted"
            record.args = ()
            return True

        # Drop all other messages from request_logger.
        return False
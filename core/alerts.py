from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="infesights-webhook")


def dispatch_webhook(
	event: dict[str, Any],
	endpoint: str | None = None,
) -> Future[str]:
	target = endpoint or os.getenv("INFESIGHTS_WEBHOOK_URL", "")
	if not target:
		future: Future[str] = Future()
		future.set_result("Webhook not configured; alert retained in the local incident log.")
		return future

	return _executor.submit(_post_webhook, event, target)


def _post_webhook(event: dict[str, Any], target: str) -> str:
	request = Request(
		target,
		data=json.dumps(event, default=str).encode("utf-8"),
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	try:
		with urlopen(request, timeout=5) as response:
			if 200 <= response.status < 300:
				return f"Webhook delivered (HTTP {response.status})."
			return f"Webhook rejected (HTTP {response.status})."
	except (OSError, URLError, ValueError) as error:
		return f"Webhook delivery failed: {error}"
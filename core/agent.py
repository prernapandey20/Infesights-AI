import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv(override=True)


class RootCauseAgent:
	PRIMARY_MODEL = "llama-3.3-70b-versatile"
	FALLBACK_MODEL = "llama-3.1-8b-instant"
	AVAILABLE_FALLBACK_MODEL = "openai/gpt-oss-20b"

	def __init__(self) -> None:
		self.model = self.PRIMARY_MODEL
		self.api_key = ""
		self.client = None
		self._refresh_configuration()

	def diagnose(
		self,
		anomaly_event: dict[str, Any],
		baseline_stats: dict[str, Any],
		recent_context_df: Any,
	) -> str:
		self._refresh_configuration()
		if self.client is None:
			return "Diagnostic agent unavailable: GROQ_API_KEY is not configured."

		context = self._format_context(recent_context_df)
		user_prompt = f"""
Incident timestamp:
{anomaly_event.get('timestamp', 'Not provided')}

Detected anomalous features and values:
{json.dumps(anomaly_event, indent=2, default=str)}

Baseline running averages and standard deviations:
{json.dumps(baseline_stats, indent=2, default=str)}

Recent telemetry context:
{context}

Computed Z-scores or deviation deltas should be used to identify the triggering metric.
""".strip()

		system_prompt = """
You are a Principal Site Reliability Engineer and Data Systems Architect.
Analyze the telemetry evidence, reason about metric interdependencies, and identify
the most likely root cause without inventing unavailable facts.
Return only clean, structured Markdown matching this exact format:
### 🚨 INCIDENT DIAGNOSIS REPORT
- **Severity:** [CRITICAL / HIGH / MEDIUM]
- **Confidence:** [X]%
- **Triggering Metric:** [Name and delta]
#### 🔍 Root-Cause Analysis
[2-3 concise sentences detailing why this spike occurred based on metric interdependencies]
#### 💡 Recommended Mitigation Steps
1. [Immediate triage action]
2. [Configuration/scaling fix]
3. [Preventative monitoring policy]
""".strip()

		try:
			return self._complete(system_prompt, user_prompt, self.PRIMARY_MODEL)
		except Exception as error:
			if isinstance(error, RateLimitError) or self._is_model_unavailable(error):
				last_error = error
				for model in (self.FALLBACK_MODEL, self.AVAILABLE_FALLBACK_MODEL):
					try:
						return self._complete(system_prompt, user_prompt, model)
					except Exception as fallback_error:
						last_error = fallback_error
				return f"Diagnostic agent error: {last_error}"
			return f"Diagnostic agent error: {error}"

	def _complete(self, system_prompt: str, user_prompt: str, model: str) -> str:
		response = self.client.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			temperature=0.2,
		)
		content = response.choices[0].message.content
		return content.strip() if content else "Diagnostic agent returned an empty response."

	@staticmethod
	def _format_context(recent_context_df: Any) -> str:
		if recent_context_df is None:
			return "No recent telemetry context provided."
		if hasattr(recent_context_df, "tail") and hasattr(recent_context_df, "to_dict"):
			recent_context_df = recent_context_df.tail(20).to_dict(orient="records")
		return json.dumps(recent_context_df, indent=2, default=str)

	def _refresh_configuration(self) -> None:
		load_dotenv(override=True)
		api_key = os.getenv("GROQ_API_KEY", "")
		if api_key != self.api_key:
			self.api_key = api_key
			self.client = Groq(api_key=api_key) if api_key else None

	@staticmethod
	def _is_model_unavailable(error: Exception) -> bool:
		message = str(error).lower()
		return "model" in message and (
			"not found" in message or "does not exist" in message
		)

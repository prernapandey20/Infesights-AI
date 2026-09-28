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
		sql_results: Any = None,
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

DuckDB investigation results:
{self._format_context(sql_results)}

Computed Z-scores or deviation deltas should be used to identify the triggering metric.
Treat query results as evidence, not proof of causality.
""".strip()

		system_prompt = """
You are an executive data analyst and causal-investigation specialist.
Infer the dataset's likely domain from its column names, values, and supplied context.
Analyze relationships among the anomalous features and identify a plausible root cause.
The data may describe e-commerce, finance, supply chains, IoT, or another domain.
Do not assume infrastructure metrics, invent business context, or state correlation as
proof of causation. Separate observed evidence from hypotheses and tailor next actions
to the inferred domain. Return concise Markdown in this format:
### Executive Anomaly Brief
- **Likely data domain:** [domain and confidence, or unknown]
- **Business impact:** [evidence-based impact or what remains unknown]
- **Anomalous features:** [names, values, and deviations]
#### Root-Cause Assessment
[Evidence, plausible explanation, and uncertainty]
#### Severity and Confidence
- **Severity:** [CRITICAL / HIGH / MEDIUM / LOW, evidence-based]
- **Confidence:** [percent and basis]
#### SQL Investigation & Audit
- **Query:** [read-only query used, or state that SQL investigation was unavailable]
- **Finding:** [what returned rows support, or limitations]
#### Recommended Actions
1. [Immediate validation or containment]
2. [Domain-specific mitigation playbook]
3. [Preventative control or monitoring]
""".strip()

		try:
			return self._complete(system_prompt, user_prompt, self.PRIMARY_MODEL)
		except Exception as error:
			if self._is_auth_error(error):
				return "Diagnostic agent unavailable: Groq rejected GROQ_API_KEY. Verify the configured API key."
			if isinstance(error, RateLimitError) or self._is_model_unavailable(error):
				last_error = error
				for model in (self.FALLBACK_MODEL, self.AVAILABLE_FALLBACK_MODEL):
					try:
						return self._complete(system_prompt, user_prompt, model)
					except Exception as fallback_error:
						last_error = fallback_error
				return f"Diagnostic agent error: {last_error}"
			return f"Diagnostic agent error: {error}"

	def generate_investigation_query(
		self,
		anomaly_event: dict[str, Any],
		column_names: list[str],
	) -> str:
		self._refresh_configuration()
		if self.client is None:
			return ""

		system_prompt = """
You are a DuckDB analyst. Return exactly one read-only SQL SELECT query and no
Markdown fences. The only available table is recent_event_rows(timestamp TIMESTAMP,
row_json JSON). Extract a numeric field with
CAST(json_extract(row_json, '$.\"field_name\"') AS DOUBLE), and extract a text field
with json_extract_string(row_json, '$.\"field_name\"'). Use only fields from the
provided schema. This table contains only the most recent 500 records. Investigate
co-occurring categories and relationships relevant to the anomaly. Include
FROM recent_event_rows and a LIMIT of at most 100 rows. Never write data or read
external files.
""".strip()
		user_prompt = (
			f"Available data columns: {json.dumps(column_names)}\n"
			f"Anomaly to investigate:\n{json.dumps(anomaly_event, indent=2, default=str)}"
		)
		try:
			response = self._complete(system_prompt, user_prompt, self.PRIMARY_MODEL)
		except Exception as error:
			if not (isinstance(error, RateLimitError) or self._is_model_unavailable(error)):
				return ""
			for model in (self.FALLBACK_MODEL, self.AVAILABLE_FALLBACK_MODEL):
				try:
					response = self._complete(system_prompt, user_prompt, model)
					break
				except Exception:
					response = ""
		return self._strip_code_fence(response)

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

	@staticmethod
	def _strip_code_fence(response: str) -> str:
		cleaned = response.strip()
		if cleaned.startswith("```"):
			lines = cleaned.splitlines()
			if lines[-1].strip().startswith("```"):
				cleaned = "\n".join(lines[1:-1])
		return cleaned.strip()

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

	@staticmethod
	def _is_auth_error(error: Exception) -> bool:
		message = str(error).lower()
		return "invalid_api_key" in message or "unauthorized" in message or "401" in message

import os

import pandas as pd
from dotenv import load_dotenv

from core.agent import RootCauseAgent


load_dotenv()

agent = RootCauseAgent()

# Simulated event based on your Tick 3 spike
dummy_event = {
    "timestamp": "2026-09-26 15:30:00",
    "feature": "api_latency_ms",
    "value": 1420.5,
    "confidence": 96.4,
}

dummy_baseline = {
    "api_latency_ms_mean": 120.0,
    "api_latency_ms_std": 15.2,
    "cpu_utilization_pct_mean": 42.0,
    "error_rate_pct_mean": 0.05,
}

print("Querying Groq Llama 3 Agent...")
report = agent.diagnose(dummy_event, dummy_baseline, pd.DataFrame())
print("\n" + report)
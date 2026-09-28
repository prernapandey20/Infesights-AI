import os

import pandas as pd
from dotenv import load_dotenv

from core.agent import RootCauseAgent


load_dotenv()

agent = RootCauseAgent()

# Simulated event based on your Tick 3 spike
dummy_event = {
    "timestamp": "2026-09-26 15:30:00",
    "features": {
        "order_value": {"value": 1420.5, "deviation_standard_deviations": 6.4},
        "fraud_score": {"value": 0.96, "deviation_standard_deviations": 5.8},
    },
    "confidence": 96.4,
}

dummy_baseline = {
    "order_value": {"mean": 120.0, "standard_deviation": 15.2},
    "fraud_score": {"mean": 0.05, "standard_deviation": 0.02},
}

print("Querying Groq Llama 3 Agent...")
report = agent.diagnose(dummy_event, dummy_baseline, pd.DataFrame())
print("\n" + report)
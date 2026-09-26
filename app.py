import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.agent import RootCauseAgent
from core.detector import AnomalyDetector
from core.storage import TelemetryStore
from core.stream_producer import StreamProducer


st.set_page_config(
	page_title="Infesights AI — Real-Time Telemetry & Agentic RCA",
	layout="wide",
)


METRIC_LABELS = {
	"cpu_utilization_pct": ("CPU Utilization", "%"),
	"memory_utilization_pct": ("Memory Utilization", "%"),
	"api_latency_ms": ("API Latency", " ms"),
	"error_rate_pct": ("Error Rate", "%"),
}


def create_telemetry_chart(
	df: pd.DataFrame,
	metric_col: str,
	title: str,
	unit: str = "",
) -> go.Figure:
	fig = go.Figure()
	fig.add_trace(
		go.Scatter(
			x=df["timestamp"],
			y=df[metric_col],
			mode="lines",
			name=metric_col,
			line=dict(color="#00d2ff", width=2),
		)
	)

	if "is_anomaly" in df.columns:
		anomaly_df = df[df["is_anomaly"] == True]
		if not anomaly_df.empty:
			fig.add_trace(
				go.Scatter(
					x=anomaly_df["timestamp"],
					y=anomaly_df[metric_col],
					mode="markers",
					name="Anomaly Detected",
					marker=dict(
						color="#ff2a5f",
						size=10,
						symbol="diamond",
						line=dict(width=1, color="white"),
					),
				)
			)

	fig.update_layout(
		title=f"<b>{title}</b>",
		yaxis_title=unit,
		margin=dict(l=10, r=10, t=35, b=10),
		height=280,
		template="plotly_dark",
		showlegend=False,
	)
	return fig


def initialize_session() -> None:
	if "inject_spike" not in st.session_state:
		st.session_state.inject_spike = None
	if "latest_report" not in st.session_state:
		st.session_state.latest_report = None
	if "anomaly_points" not in st.session_state:
		st.session_state.anomaly_points = {}
	if "producer" not in st.session_state:
		st.session_state.producer = StreamProducer()
		st.session_state.detector = AnomalyDetector()
		st.session_state.store = TelemetryStore()
		st.session_state.agent = RootCauseAgent()
		st.session_state.streaming_active = False
		st.session_state.latest_point = None
		st.session_state.latest_result = None
		st.session_state.latest_report = None

		warmup = [st.session_state.producer.get_next_point() for _ in range(50)]
		st.session_state.detector.fit_initial_baseline(warmup)
		for point in warmup:
			save_point(point)
		st.session_state.latest_point = warmup[-1]


def save_point(point: dict) -> None:
	producer = st.session_state.producer
	store = st.session_state.store
	for feature_name in METRIC_LABELS:
		store.insert_telemetry(
			point["timestamp"],
			feature_name,
			float(point[feature_name]),
			producer.rolling_means[feature_name],
		)


def process_next_point() -> None:
	producer = st.session_state.producer
	detector = st.session_state.detector
	store = st.session_state.store
	point = producer.get_next_point()
	result = detector.predict(point)
	print(
		f"[STREAM TICK] Latency: {point.get('api_latency_ms')} | "
		f"Anomaly: {result.get('is_anomaly')}"
	)
	save_point(point)
	st.session_state.latest_point = point
	st.session_state.latest_result = result

	if result["is_anomaly"]:
		metric_name = max(result["deviations"], key=result["deviations"].get)
		st.session_state.anomaly_points.setdefault(metric_name, set()).add(
			point["timestamp"]
		)
		baseline_stats = {
			f"{name}_mean": detector.baseline_means[name]
			for name in detector.feature_names
		}
		baseline_stats.update(
			{
				f"{name}_std": detector.baseline_stds[name]
				for name in detector.feature_names
			}
		)
		anomaly_event = {
			"timestamp": point["timestamp"],
			"feature": metric_name,
			"value": point[metric_name],
			"confidence": result["confidence"],
			"deviations": result["deviations"],
		}
		st.toast("Anomaly Detected by Isolation Forest!", icon="⚠️")
		store.log_incident(
			metric_name,
			float(point[metric_name]),
			detector.baseline_means[metric_name],
			result["confidence"],
		)
		with st.spinner("AI Agent diagnosing root cause..."):
			st.session_state.latest_report = st.session_state.agent.diagnose(
				anomaly_event,
				baseline_stats,
				store.get_recent_telemetry(limit=20),
			)

initialize_session()

with st.sidebar:
	st.title("Infesights AI")
	st.caption("Real-time telemetry and agentic root-cause analysis")
	st.toggle("Start / Stop Stream", key="streaming_active")
	interval = st.slider("Streaming interval", 0.2, 2.0, 0.8, 0.1, format="%.1fs")
	st.divider()
	st.subheader("Chaos Controls")
	if st.button("⚡ Inject Latency Spike"):
		st.session_state.inject_spike = "api_latency_ms"
	if st.button("💥 Inject Error Surge"):
		st.session_state.inject_spike = "error_rate_pct"

if st.session_state.streaming_active:
	if st.session_state.inject_spike:
		target_feature = st.session_state.inject_spike
		st.session_state.producer.inject_anomaly(
			feature=target_feature,
			magnitude=10.0,
		)
		st.session_state.inject_spike = None
	process_next_point()

st.title("Infesights AI — Real-Time Telemetry & Agentic RCA")
st.caption("Synthetic infrastructure telemetry with autonomous anomaly diagnosis")

point = st.session_state.latest_point
if point is not None:
	metric_columns = st.columns(4)
	for column, (feature, (label, suffix)) in zip(metric_columns, METRIC_LABELS.items()):
		column.metric(label, f"{point[feature]:.2f}{suffix}")

if st.session_state.get("latest_report"):
	with st.container():
		st.error("🚨 Active Anomaly Detected — Autonomous Root-Cause Diagnosis")
		with st.expander("📄 View Incident Post-Mortem & Remediation Steps", expanded=True):
			st.markdown(st.session_state.latest_report)
			col_act1, col_act2 = st.columns(2)
			with col_act1:
				if st.button("🛡️ Execute Mitigation: Apply Rate Limiting"):
					st.success("Policy dispatched: Endpoint throttled to 200 req/sec.")
			with col_act2:
				if st.button("🔄 Resolve Incident & Dismiss"):
					st.session_state.latest_report = None
					st.rerun()

telemetry = st.session_state.store.get_recent_telemetry(limit=400)
if not telemetry.empty:
	recent_df = telemetry.sort_values("timestamp").tail(60)
	st.subheader("Live Telemetry")
	chart_columns = st.columns(2)
	for column, feature in zip(chart_columns * 2, METRIC_LABELS):
		metric_data = recent_df[recent_df["feature_name"] == feature].copy()
		if not metric_data.empty:
			metric_data["is_anomaly"] = metric_data["timestamp"].isin(
				st.session_state.anomaly_points.get(feature, set())
			)
			figure = create_telemetry_chart(
				metric_data,
				"metric_value",
				METRIC_LABELS[feature][0],
				METRIC_LABELS[feature][1],
			)
			column.plotly_chart(figure)

st.markdown("---")
st.subheader("📑 Incident Audit History")
store = st.session_state.store
if hasattr(store, "get_open_incidents"):
	incidents_df = store.get_open_incidents()
	if not incidents_df.empty:
		st.dataframe(incidents_df, width="stretch")
		csv_bytes = incidents_df.to_csv(index=False).encode("utf-8")
		st.download_button(
			label="📥 Export Incident Log (CSV)",
			data=csv_bytes,
			file_name="infesights_incidents.csv",
			mime="text/csv",
		)
	else:
		st.info("System healthy: No open incidents recorded.")

if st.session_state.streaming_active:
	time.sleep(interval)
	st.rerun()

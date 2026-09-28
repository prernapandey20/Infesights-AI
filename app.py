import hashlib
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.alerts import dispatch_webhook
from core.agent import RootCauseAgent
from core.data_pipeline import create_demo_dataset, load_dataset, prepare_dataset
from core.detector import AnomalyDetector
from core.drift import compare_distributions
from core.report_export import create_executive_pdf
from core.storage import TelemetryStore
from core.stream_producer import StreamProducer


def create_telemetry_chart(
	df: pd.DataFrame,
	metric_col: str,
	title: str,
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
		margin=dict(l=10, r=10, t=35, b=10),
		height=280,
		template="plotly_dark",
		showlegend=False,
	)
	return fig


def anomalous_features(result: dict) -> list[str]:
	features = [
		name
		for name, deviation in result["deviations"].items()
		if deviation >= result.get("deviation_threshold", 2.5)
	]
	if result["is_anomaly"] and not features:
		features = [max(result["deviations"], key=result["deviations"].get)]
	return features


def initialize_dataset(
	frame: pd.DataFrame,
	features: list[str],
	signature: str,
) -> None:
	required_state = {
		"baseline_frame",
		"incoming_records",
		"latest_sql_result",
		"latest_anomaly_event",
		"alert_status",
		"alert_future",
	}
	if (
		st.session_state.get("dataset_signature") == signature
		and required_state.issubset(st.session_state.keys())
	):
		return

	records = frame.to_dict(orient="records")
	baseline_end = max(1, min(len(records) - 1, int(len(records) * 0.7)))
	baseline_records = records[:baseline_end]
	detector = AnomalyDetector()
	detector.fit_initial_baseline(baseline_records)
	producer = StreamProducer(records, features)
	store = TelemetryStore()
	anomaly_points = {feature: set() for feature in features}
	rolling_values = {feature: [] for feature in features}
	for record in records:
		store.insert_event(record)

	start_index = max(0, len(records) - 400)
	for record_index in range(start_index, len(records)):
		record = records[record_index]
		result = detector.predict(record) if record_index >= baseline_end else None
		flagged_features = anomalous_features(result) if result else []
		for feature in features:
			value = float(record[feature])
			rolling_values[feature].append(value)
			store.insert_telemetry(
				record["timestamp"],
				feature,
				value,
				sum(rolling_values[feature]) / len(rolling_values[feature]),
			)
			if feature in flagged_features:
				anomaly_points[feature].add(record["timestamp"])
				store.log_incident(
					feature,
					value,
					detector.baseline_means[feature],
					result["confidence"],
				)

	st.session_state.dataset_signature = signature
	st.session_state.features = features
	st.session_state.producer = producer
	st.session_state.detector = detector
	st.session_state.store = store
	st.session_state.agent = RootCauseAgent()
	st.session_state.anomaly_points = anomaly_points
	st.session_state.baseline_frame = pd.DataFrame(baseline_records)
	st.session_state.incoming_records = records[baseline_end:][-1000:]
	st.session_state.latest_point = records[-1]
	st.session_state.latest_result = None
	st.session_state.latest_report = None
	st.session_state.latest_sql_query = ""
	st.session_state.latest_sql_result = None
	st.session_state.latest_anomaly_event = None
	st.session_state.latest_anomaly_point = None
	st.session_state.last_alert_id = None
	st.session_state.alert_status = ""
	st.session_state.alert_future = None
	st.session_state.counterfactual_result = None
	st.session_state.streaming_active = False


def save_point(point: dict) -> None:
	st.session_state.store.insert_event(point)
	st.session_state.incoming_records.append(point.copy())
	st.session_state.incoming_records = st.session_state.incoming_records[-1000:]
	for feature in st.session_state.features:
		st.session_state.store.insert_telemetry(
			point["timestamp"],
			feature,
			float(point[feature]),
			st.session_state.producer.rolling_means[feature],
		)


def investigate_event(
	anomaly_event: dict,
	baseline_stats: dict,
) -> str:
	store = st.session_state.store
	agent = st.session_state.agent
	query = agent.generate_investigation_query(
		anomaly_event,
		list(st.session_state.latest_point.keys()),
	)
	query_result: dict = {"query": query, "rows": []}
	if query:
		try:
			result_frame = store.execute_readonly_query(query)
			query_result["rows"] = result_frame.to_dict(orient="records")
		except Exception as error:
			query_result["error"] = str(error)
	st.session_state.latest_sql_query = query
	st.session_state.latest_sql_result = query_result if query else None
	return agent.diagnose(
		anomaly_event,
		baseline_stats,
		store.get_recent_telemetry(limit=40),
		sql_results=query_result if query else None,
	)


def process_next_point() -> None:
	producer = st.session_state.producer
	detector = st.session_state.detector
	store = st.session_state.store
	point = producer.get_next_point()
	result = detector.predict(point)
	save_point(point)
	st.session_state.latest_point = point
	st.session_state.latest_result = result

	flagged_features = anomalous_features(result)
	if not flagged_features:
		return

	for feature in flagged_features:
		st.session_state.anomaly_points[feature].add(point["timestamp"])
		store.log_incident(
			feature,
			float(point[feature]),
			detector.baseline_means[feature],
			result["confidence"],
		)
	anomaly_event = {
		"timestamp": point["timestamp"],
		"features": {
			feature: {
				"value": point[feature],
				"deviation_standard_deviations": result["deviations"][feature],
			}
			for feature in flagged_features
		},
		"confidence": result["confidence"],
		"row_context": {
			key: value
			for key, value in point.items()
			if key not in st.session_state.features and key != "timestamp"
		},
	}
	st.session_state.latest_anomaly_event = anomaly_event
	st.session_state.latest_anomaly_point = point.copy()
	baseline_stats = {
		feature: {
			"mean": detector.baseline_means[feature],
			"standard_deviation": detector.baseline_stds[feature],
		}
		for feature in detector.feature_names
	}
	st.toast("Anomaly detected in the ingested data", icon="⚠️")
	with st.spinner("Preparing an evidence-based data-domain assessment..."):
		st.session_state.latest_report = investigate_event(anomaly_event, baseline_stats)
	threshold = result["alert_threshold"]
	if result["confidence"] >= threshold:
		alert_id = str(point["timestamp"])
		if st.session_state.last_alert_id != alert_id:
			st.session_state.alert_future = dispatch_webhook(
				{
					"event": "anomaly_detected",
					"confidence": result["confidence"],
					"threshold": threshold,
					"anomaly": anomaly_event,
				}
			)
			st.session_state.alert_status = "Webhook delivery queued asynchronously."
			st.session_state.last_alert_id = alert_id


def reevaluate_counterfactual(values: dict[str, float]) -> None:
	detector = st.session_state.detector
	point = st.session_state.latest_anomaly_point.copy()
	point.update(values)
	result = detector.predict(point, update_buffer=False)
	event = {
		**st.session_state.latest_anomaly_event,
		"counterfactual_values": values,
		"counterfactual_risk": {
			"is_anomaly": result["is_anomaly"],
			"confidence": result["confidence"],
			"deviations": result["deviations"],
		},
	}
	baseline_stats = {
		feature: {
			"mean": detector.baseline_means[feature],
			"standard_deviation": detector.baseline_stds[feature],
		}
		for feature in detector.feature_names
	}
	st.session_state.latest_report = investigate_event(event, baseline_stats)
	st.session_state.counterfactual_result = result


st.set_page_config(
	page_title="Infesights AI — Universal Data Intelligence",
	layout="wide",
)
st.title("Infesights AI — Universal Data Intelligence")
st.caption("Investigate, monitor, and explain anomalies in any tabular dataset")

st.sidebar.title("Infesights AI")
st.sidebar.caption("Dynamic Schema Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload CSV or JSON", type=["csv", "json"])

try:
	if uploaded_file is None:
		if "demo_dataset" not in st.session_state:
			st.session_state.demo_dataset = create_demo_dataset()
		source_frame = st.session_state.demo_dataset
		signature = "demo-orders-v1"
	else:
		file_data = uploaded_file.getvalue()
		source_frame = load_dataset(file_data, uploaded_file.name)
		signature = hashlib.sha256(file_data).hexdigest()
	prepared_frame, numeric_features = prepare_dataset(source_frame)
	if len(prepared_frame) < 2:
		st.error("Upload at least two records so the baseline can be estimated.")
		st.stop()
except Exception as error:
	st.error(f"Unable to load dataset: {error}")
	st.stop()

initialize_dataset(prepared_frame, numeric_features, signature)

with st.sidebar.expander("⚙️ Streaming & Replay Controls", expanded=True):
	st.toggle("Live stream mode", key="streaming_active")
	st.caption(
		"Static uploads run batch detection immediately. Live mode ingests rows automatically."
	)
	interval = 0.8

with st.sidebar.expander("🔔 Webhook & Alerting", expanded=False):
	st.caption("Alert confidence is calibrated from the top 2% of baseline deviations.")
	if os.getenv("INFESIGHTS_WEBHOOK_URL"):
		st.caption("Webhook dispatcher configured")
	else:
		st.caption("Set INFESIGHTS_WEBHOOK_URL to enable notifications")

st.sidebar.caption(f"{len(prepared_frame):,} rows · {len(numeric_features)} numeric features")

streaming_active = st.session_state.streaming_active


@st.fragment(run_every=interval if streaming_active else None)
def render_dashboard() -> None:
	if st.session_state.streaming_active:
		process_next_point()

	store = st.session_state.store
	alert_future = st.session_state.alert_future
	if alert_future is not None and alert_future.done():
		st.session_state.alert_status = alert_future.result()
		st.session_state.alert_future = None
	telemetry = store.get_recent_telemetry(limit=400 * len(numeric_features))
	if not telemetry.empty:
		recent_df = telemetry.sort_values("timestamp").groupby("feature_name").tail(60)
	else:
		recent_df = telemetry

	summary_columns = st.columns(3)
	summary_columns[0].metric("Records loaded", f"{len(prepared_frame):,}")
	summary_columns[1].metric("Numeric features", len(numeric_features))
	summary_columns[2].metric("Anomaly events", len(store.get_open_incidents()))

	live_tab, drift_tab = st.tabs(["Live analysis", "Feature drift"])
	with live_tab:
		latest = st.session_state.latest_point
		if st.session_state.get("latest_report"):
			st.markdown("---")
			st.subheader("💡 Automated AI Insight & Summary")
			with st.container(border=True):
				st.markdown(st.session_state.latest_report)
				st.download_button(
					"Export Executive Post-Mortem (Markdown)",
					data=st.session_state.latest_report.encode("utf-8"),
					file_name="infesights_executive_postmortem.md",
					mime="text/markdown",
					key=f"markdown_{signature}",
				)
				st.download_button(
					"Export Executive Post-Mortem (PDF)",
					data=create_executive_pdf(st.session_state.latest_report),
					file_name="infesights_executive_postmortem.pdf",
					mime="application/pdf",
					key=f"pdf_{signature}",
				)
				if st.session_state.alert_status:
					st.caption(st.session_state.alert_status)
				if st.button("👍 Got It / Dismiss Summary"):
					st.session_state.latest_report = None
					st.rerun()

		if st.session_state.latest_sql_result:
			with st.expander("Agent SQL investigation", expanded=True):
				st.code(st.session_state.latest_sql_query, language="sql")
				if "error" in st.session_state.latest_sql_result:
					st.warning(st.session_state.latest_sql_result["error"])
				else:
					st.dataframe(
						st.session_state.latest_sql_result["rows"],
						width="stretch",
					)

		if st.session_state.latest_anomaly_event:
			with st.expander("What-if risk simulator"):
				with st.form(f"counterfactual_{signature}"):
					counterfactual_values = {}
					for feature, anomaly in st.session_state.latest_anomaly_event[
						"features"
					].items():
						mean = st.session_state.detector.baseline_means[feature]
						deviation = st.session_state.detector.baseline_stds[feature]
						current_value = float(anomaly["value"])
						span = max(abs(mean) * 0.01, deviation * 6, 1e-6)
						counterfactual_values[feature] = st.slider(
							feature.replace("_", " ").title(),
							min_value=float(min(mean - span, current_value)),
							max_value=float(max(mean + span, current_value)),
							value=current_value,
							key=f"counterfactual_{signature}_{feature}",
						)
					recheck = st.form_submit_button("Re-evaluate risk")
				if recheck:
					reevaluate_counterfactual(counterfactual_values)
				if st.session_state.get("counterfactual_result"):
					counterfactual = st.session_state.counterfactual_result
					st.metric(
						"Counterfactual anomaly risk",
						"Elevated" if counterfactual["is_anomaly"] else "Not flagged",
						f"{counterfactual['confidence']:.1f}% model score",
					)

		st.subheader("Dynamic feature streams")
		chart_columns = st.columns(2)
		for index, feature in enumerate(numeric_features):
			feature_data = recent_df[recent_df["feature_name"] == feature].copy()
			if feature_data.empty:
				continue
			feature_data["is_anomaly"] = feature_data["timestamp"].isin(
				st.session_state.anomaly_points.get(feature, set())
			)
			figure = create_telemetry_chart(
				feature_data,
				"metric_value",
				feature.replace("_", " ").title(),
			)
			chart_columns[index % 2].plotly_chart(figure, width="stretch")

		with st.expander("Latest ingested row"):
			st.json(latest, expanded=True)
		with st.expander("Dataset preview"):
			st.dataframe(prepared_frame.head(20), width="stretch")

		st.subheader("Anomaly event log")
		incidents = store.get_recent_incidents(limit=100)
		if incidents.empty:
			st.info("No anomalies detected in the loaded rows yet.")
		else:
			st.dataframe(
				incidents.rename(columns={"metric_name": "feature"}),
				width="stretch",
			)
			st.download_button(
				"Export anomaly log (CSV)",
				data=incidents.to_csv(index=False).encode("utf-8"),
				file_name="infesights_anomalies.csv",
				mime="text/csv",
			)

	with drift_tab:
		baseline_frame = st.session_state.baseline_frame
		incoming_frame = pd.DataFrame(st.session_state.incoming_records)
		st.subheader("Baseline vs incoming distribution")
		if len(incoming_frame) < 2:
			st.info("At least two incoming rows are needed to calculate distribution drift.")
		else:
			window_size = st.number_input(
				"Incoming comparison rows",
				min_value=2,
				max_value=len(incoming_frame),
				value=min(200, len(incoming_frame)),
			)
			comparison_frame = incoming_frame.tail(int(window_size))
			drift_rows = []
			for feature in numeric_features:
				statistics = compare_distributions(
					baseline_frame[feature],
					comparison_frame[feature],
				)
				psi = statistics["psi"]
				drift_rows.append(
					{
						"feature": feature,
						"KS statistic": statistics["ks_statistic"],
						"KS p-value": statistics["ks_p_value"],
						"PSI": psi,
						"PSI assessment": (
							"Stable" if psi < 0.1 else "Moderate" if psi < 0.2 else "Material"
						),
					}
				)
			st.dataframe(pd.DataFrame(drift_rows), width="stretch", hide_index=True)
			drift_feature = st.selectbox(
				"Inspect distribution",
				numeric_features,
				key=f"drift_feature_{signature}",
			)
			figure = go.Figure()
			figure.add_trace(
				go.Histogram(
					x=baseline_frame[drift_feature],
					name="Training baseline",
					histnorm="probability density",
					opacity=0.55,
				)
			)
			figure.add_trace(
				go.Histogram(
					x=comparison_frame[drift_feature],
					name="Incoming window",
					histnorm="probability density",
					opacity=0.55,
				)
			)
			figure.update_layout(
				barmode="overlay",
				title=f"{drift_feature.replace('_', ' ').title()} distribution",
				height=360,
			)
			st.plotly_chart(figure, width="stretch")


render_dashboard()

import hashlib
from io import BytesIO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.alerts import dispatch_webhook
from core.agent import RootCauseAgent
from core.data_pipeline import create_demo_dataset, prepare_dataset
from core.detector import AnomalyDetector
from core.drift import compare_distributions
from core.report_export import (
	create_executive_pdf,
	generate_excel_report,
	generate_word_report,
)
from core.storage import TelemetryStore
from core.stream_producer import StreamProducer


def apply_aeux_theme() -> None:
	st.markdown(
		"""
		<style>
		/* 1. Global Canvas */
		.stApp {
			background-color: #0b1f1a !important;
		}

		header[data-testid="stHeader"] {
			background-color: transparent !important;
		}

		[data-testid="stMain"] {
			background-color: #0b1f1a !important;
			padding: 1.5rem 2.5rem !important;
		}

		[data-testid="stMainBlockContainer"] {
			background-color: #f5f8f6 !important;
			border-radius: 28px !important;
			padding: 3rem 3.5rem !important;
			margin-top: 0.5rem !important;
			margin-bottom: 2rem !important;
			box-shadow: 0 20px 45px rgba(0, 0, 0, 0.35) !important;
		}

		/* 2. Left Sidebar */
		[data-testid="stSidebar"] {
			background-color: #0b1f1a !important;
			border-right: none !important;
		}

		[data-testid="stSidebar"] * {
			color: #d1deda !important;
		}

		/* 3. Typography */
		[data-testid="stMain"] h1,
		[data-testid="stMain"] h2,
		[data-testid="stMain"] h3,
		[data-testid="stMain"] h4,
		[data-testid="stMain"] h5,
		[data-testid="stMain"] p,
		[data-testid="stMain"] label,
		[data-testid="stMain"] div[data-testid="stMarkdownContainer"] * {
			color: #0d1a16 !important;
			font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
		}

		/* 4. Expander Container & Headers */
		div[data-testid="stExpander"] {
			background-color: #ffffff !important;
			border: 1px solid #e1e9e5 !important;
			border-radius: 16px !important;
			margin-bottom: 12px !important;
		}

		div[data-testid="stExpander"] summary {
			background-color: #f0f5f3 !important;
			border-radius: 16px !important;
			padding: 12px 18px !important;
		}

		div[data-testid="stExpander"] summary * {
			color: #0d1a16 !important;
			font-weight: 700 !important;
		}

		/* 5. FIX DOWNLOAD BUTTONS TEXT VISIBILITY */
		button, 
		.stButton > button, 
		div[data-testid="stDownloadButton"] > button,
		button[data-testid="baseButton-secondary"],
		button[data-testid="baseButton-primary"] {
			background-color: #0d1a16 !important;
			color: #ffffff !important;
			border-radius: 12px !important;
			border: 1px solid #0d1a16 !important;
			padding: 12px 24px !important;
			font-weight: 700 !important;
			font-size: 0.95rem !important;
			box-shadow: 0 4px 12px rgba(13, 26, 22, 0.25) !important;
		}

		button *, 
		.stButton > button *, 
		div[data-testid="stDownloadButton"] > button *,
		div[data-testid="stDownloadButton"] button div,
		div[data-testid="stDownloadButton"] button p,
		div[data-testid="stDownloadButton"] button span {
			color: #ffffff !important;
			fill: #ffffff !important;
			font-weight: 700 !important;
		}

		button:hover, 
		.stButton > button:hover, 
		div[data-testid="stDownloadButton"] > button:hover {
			background-color: #10b981 !important;
			border-color: #10b981 !important;
		}

		button:hover *, 
		.stButton > button:hover *, 
		div[data-testid="stDownloadButton"] > button:hover * {
			color: #0d1a16 !important;
			fill: #0d1a16 !important;
		}

		/* 6. Code & JSON Containers */
		.stCodeBlock, div[data-testid="stCodeBlock"] {
			border-radius: 12px !important;
			overflow: hidden !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


st.set_page_config(
	page_title="Infesights AI",
	layout="wide",
	initial_sidebar_state="expanded",
)
apply_aeux_theme()


@st.cache_data
def load_data(file_data: bytes, filename: str) -> pd.DataFrame | None:
	file = BytesIO(file_data)
	filename = filename.lower()
	if filename.endswith(".csv"):
		try:
			return pd.read_csv(file)
		except UnicodeDecodeError:
			file.seek(0)
			return pd.read_csv(file, encoding="ISO-8859-1")
	if filename.endswith((".xlsx", ".xls")):
		return pd.read_excel(file)
	if filename.endswith(".json"):
		return pd.read_json(file)
	return None


def create_dynamic_chart(
	df: pd.DataFrame,
	col_name: str,
	chart_type: str = "Line",
	anomaly_indices: list[int] | None = None,
) -> go.Figure:
	"""Render a Plotly chart using the actual dataset column name."""
	title_text = f"{col_name} ({chart_type} Chart)"
	labels = {col_name: col_name, "index": "Timestamp / Record"}
	if chart_type == "Bar":
		fig = px.bar(
			df,
			y=col_name,
			title=title_text,
			labels=labels,
			color_discrete_sequence=["#10b981"],
		)
	elif chart_type == "Area":
		fig = px.area(
			df,
			y=col_name,
			title=title_text,
			labels=labels,
			color_discrete_sequence=["#10b981"],
		)
	else:
		fig = px.line(
			df,
			y=col_name,
			title=title_text,
			labels=labels,
			color_discrete_sequence=["#0d1a16"],
		)

	fig.update_layout(
		paper_bgcolor="rgba(255,255,255,1)",
		plot_bgcolor="rgba(245,248,246,0.6)",
		font=dict(color="#0d1a16", family="Inter"),
		title=dict(font=dict(color="#0d1a16", size=14, family="Inter")),
		yaxis=dict(
			title=dict(text=col_name, font=dict(color="#0d1a16")),
			tickfont=dict(color="#0d1a16"),
			gridcolor="#e2ebe6",
			zerolinecolor="#d1deda",
		),
		xaxis=dict(
			title=dict(text="Time / Order Index", font=dict(color="#0d1a16")),
			tickfont=dict(color="#0d1a16"),
			gridcolor="#e2ebe6",
		),
		margin=dict(l=20, r=20, t=40, b=20),
		height=280,
		showlegend=False,
	)

	anomaly_indices = anomaly_indices or []
	if anomaly_indices:
		anomaly_frame = df.iloc[anomaly_indices]
		fig.add_trace(
			go.Scatter(
				x=anomaly_frame.index,
				y=anomaly_frame[col_name],
				mode="markers",
				marker=dict(color="#ef4444", size=10, symbol="diamond"),
				name="Anomaly Event",
			)
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


def ensure_ai_summary(
	dataset_name: str,
	records_count: int,
	anomaly_count: int,
) -> None:
	if anomaly_count <= 0 or "ai_summary_text" in st.session_state:
		return

	with st.spinner("Generating plain-English executive summary..."):
		summary_prompt = (
			"Write a crisp, 3-bullet-point executive summary for a non-technical "
			f"business user. The dataset '{dataset_name}' has {records_count} "
			f"records and {anomaly_count} flagged anomalies. Explain in everyday "
			"language what metrics spiked, why it matters, and what action to take."
		)
		st.session_state.ai_summary_text = st.session_state.agent.diagnose_simple(
			summary_prompt
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
	ensure_ai_summary(
		st.session_state.dataset_name,
		st.session_state.records_count,
		len(store.get_open_incidents()),
	)
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


st.sidebar.title("Infesights AI")
st.sidebar.caption("Universal Data & Anomaly Intelligence")
uploaded_file = st.sidebar.file_uploader(
	"Upload Dataset",
	type=["csv", "xlsx", "xls", "json"],
	help="Support for CSV, Excel (.xlsx), and JSON tabular datasets.",
)
is_live_stream = st.sidebar.toggle("Enable Live Stream Mode", value=False)

if uploaded_file is None and not is_live_stream:
	st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)
	st.markdown(
		"<div class='sub-caption'>Universal Data Intelligence & Automated Anomaly Detection</div>",
		unsafe_allow_html=True,
	)

	st.markdown("""
        <div class='welcome-card'>
            👋 <b>Welcome to Infesights AI</b><br><br>
            Upload a <b>CSV, Excel, or JSON</b> dataset in the left sidebar to generate real-time metrics, interactive charts, and plain-English executive reports.
        </div>
    """, unsafe_allow_html=True)
	st.stop()

st.title("Infesights AI — Universal Data Intelligence")
st.caption("Investigate, monitor, and explain anomalies in any tabular dataset")

try:
	if uploaded_file is None:
		if "demo_dataset" not in st.session_state:
			st.session_state.demo_dataset = create_demo_dataset()
		source_frame = st.session_state.demo_dataset
		signature = "demo-orders-v1"
	else:
		file_data = uploaded_file.getvalue()
		source_frame = load_data(file_data, uploaded_file.name)
		if source_frame is None:
			raise ValueError("Upload a CSV, Excel, or JSON file")
		signature = hashlib.sha256(file_data).hexdigest()
	prepared_frame, numeric_features = prepare_dataset(source_frame)
	if len(prepared_frame) < 2:
		st.error("Upload at least two records so the baseline can be estimated.")
		st.stop()
except Exception as error:
	st.error(f"Unable to load dataset: {error}")
	st.stop()

initialize_dataset(prepared_frame, numeric_features, signature)
st.session_state.dataset_name = uploaded_file.name if uploaded_file else "Live Demo Dataset"
st.session_state.records_count = len(prepared_frame)
ensure_ai_summary(
	st.session_state.dataset_name,
	st.session_state.records_count,
	len(st.session_state.store.get_open_incidents()),
)

interval = 0.8

st.session_state.streaming_active = is_live_stream
st.sidebar.caption(f"{len(prepared_frame):,} rows · {len(numeric_features)} numeric features")

streaming_active = is_live_stream


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

	st.markdown("<br>", unsafe_allow_html=True)
	anomaly_count = len(store.get_open_incidents())
	anomaly_rate = min(anomaly_count / max(len(prepared_frame), 1), 1.0)
	middle_columns = st.columns([1, 1, 2])
	with middle_columns[0]:
		st.subheader("Anomaly Load")
		st.metric("Flagged records", f"{anomaly_count:,}")
		st.progress(anomaly_rate)
		st.caption(f"{anomaly_rate:.1%} of loaded records flagged")

	with middle_columns[1]:
		st.subheader("Stream Health")
		st.metric(
			"Status",
			"Live" if st.session_state.streaming_active else "Ready",
		)
		st.caption(f"Monitoring {len(numeric_features)} numeric metrics")
		st.caption("🟢 Detector baseline calibrated")

	with middle_columns[2]:
		st.subheader("Feature Baseline Overview")
		baseline_values = prepared_frame[numeric_features].mean().sort_values(ascending=False)
		baseline_figure = px.bar(
			x=baseline_values.index,
			y=baseline_values.values,
			labels={"x": "Metric", "y": "Baseline average"},
			color_discrete_sequence=["#20e070"],
		)
		baseline_figure.update_layout(
			paper_bgcolor="rgba(0,0,0,0)",
			plot_bgcolor="rgba(0,0,0,0)",
			margin=dict(l=10, r=10, t=20, b=20),
			height=230,
		)
		st.plotly_chart(baseline_figure, width="stretch")

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
		numeric_cols = prepared_frame.select_dtypes(include=["number"]).columns.tolist()
		numeric_cols = [
			column
			for column in numeric_cols
			if not any(
				id_word in column.lower() for id_word in ["id", "code", "zip"]
			)
		]
		chart_type = st.selectbox(
			"Chart type",
			["Line", "Bar", "Area"],
			key=f"chart_type_{signature}",
		)
		chart_columns = st.columns(2)
		for index, column in enumerate(numeric_cols):
			with chart_columns[index % 2]:
				feature_data = recent_df[
					recent_df["feature_name"] == column
				].copy()
				if feature_data.empty:
					continue
				feature_data.reset_index(drop=True, inplace=True)
				feature_data["is_anomaly"] = feature_data["timestamp"].isin(
					st.session_state.anomaly_points.get(column, set())
				)
				anomaly_indices = feature_data.index[
					feature_data["is_anomaly"]
				].tolist()
				chart_data = feature_data.rename(columns={"metric_value": column})
				figure = create_dynamic_chart(
					chart_data,
					col_name=column,
					chart_type=chart_type,
					anomaly_indices=anomaly_indices,
				)
				st.plotly_chart(figure, width="stretch")

		with st.expander("Latest ingested row"):
			import json
			formatted_json = json.dumps(latest, indent=4, default=str)
			st.code(formatted_json, language="json")
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

		st.markdown("---")
		st.subheader("Download Reports")
		col_dl1, col_dl2 = st.columns(2)
		anomaly_df = incidents
		ai_summary_text = st.session_state.get("ai_summary_text") or st.session_state.latest_report or (
			"No AI anomaly summary has been generated yet."
		)

		with col_dl1:
			excel_bytes = generate_excel_report(prepared_frame, anomaly_df)
			st.download_button(
				label="📊 Download Excel Report (.xlsx)",
				data=excel_bytes,
				file_name="Infesights_Data_Report.xlsx",
				mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
				key=f"analysis_excel_{signature}",
			)

		with col_dl2:
			word_bytes = generate_word_report(ai_summary_text, anomaly_df)
			st.download_button(
				label="📄 Download Executive Word Report (.docx)",
				data=word_bytes,
				file_name="Infesights_Executive_Report.docx",
				mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
				key=f"analysis_word_{signature}",
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

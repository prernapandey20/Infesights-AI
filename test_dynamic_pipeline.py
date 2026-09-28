import unittest
from datetime import datetime, timedelta
from threading import Event
from unittest.mock import patch

import pandas as pd

from core.alerts import dispatch_webhook
from core.agent import RootCauseAgent
from core.data_pipeline import load_dataset, prepare_dataset
from core.detector import AnomalyDetector
from core.drift import compare_distributions
from core.report_export import create_executive_pdf
from core.storage import TelemetryStore
from core.stream_producer import StreamProducer


class DynamicPipelineTests(unittest.TestCase):
	def test_csv_and_json_infer_numeric_features(self) -> None:
		csv_data = (
			b"timestamp,order_value,region\n"
			b"2026-01-01T00:00:00,12.5,north\n"
			b"2026-01-01T00:05:00,15.0,south\n"
		)
		json_data = b'[{"timestamp":"2026-01-01T00:00:00","yield_rate":0.8,"site":"a"}]'

		csv_frame, csv_features = prepare_dataset(load_dataset(csv_data, "orders.csv"))
		json_frame, json_features = prepare_dataset(load_dataset(json_data, "yield.json"))

		self.assertEqual(csv_features, ["order_value"])
		self.assertEqual(json_features, ["yield_rate"])
		self.assertEqual(len(csv_frame), 2)
		self.assertEqual(len(json_frame), 1)

	def test_legacy_csv_falls_back_to_latin1(self) -> None:
		frame = load_dataset(b"name\nAndr\xe9\n", "legacy.csv")

		self.assertEqual(frame.loc[0, "name"], "Andr\u00e9")

	def test_datetime_columns_are_not_numeric_features(self) -> None:
		frame = pd.DataFrame(
			{
				"timestamp": pd.date_range("2026-01-01", periods=2, freq="h"),
				"created_at": pd.date_range("2025-12-01", periods=2, freq="D"),
				"order_total": [12.5, 20.0],
			}
		)

		_, features = prepare_dataset(frame)

		self.assertEqual(features, ["order_total"])

	def test_detector_and_stream_work_with_arbitrary_features(self) -> None:
		records = [
			{
				"timestamp": datetime(2026, 1, 1) + timedelta(minutes=index),
				"transaction_total": float(index),
				"risk_score": index / 100,
			}
			for index in range(50)
		]
		detector = AnomalyDetector()
		detector.fit_initial_baseline(records)
		producer = StreamProducer(records, ["transaction_total", "risk_score"])

		producer.inject_anomaly("risk_score", magnitude=8.0)
		point = producer.get_next_point()
		result = detector.predict(point)

		self.assertEqual(set(result["deviations"]), {"transaction_total", "risk_score"})
		self.assertEqual(len(producer.history), 1)
		self.assertGreater(point["risk_score"], 0.49)
		self.assertGreater(point["timestamp"], point["event_timestamp"])
		buffer_size = len(detector.buffer)
		detector.predict({**point, "risk_score": 0.01}, update_buffer=False)
		self.assertEqual(len(detector.buffer), buffer_size)

	def test_duckdb_query_tool_is_read_only_and_scoped(self) -> None:
		store = TelemetryStore()
		store.insert_event(
			{"timestamp": datetime.now(), "transaction_amount": 150.0, "user_location": "north"}
		)
		result = store.execute_readonly_query(
			"SELECT json_extract_string(row_json, '$.user_location') AS location "
			"FROM recent_event_rows"
		)

		self.assertEqual(result.loc[0, "location"], "north")
		for query in (
			"DELETE FROM event_rows",
			"SELECT 1 FROM event_rows",
			"SELECT 1; SELECT 2 FROM recent_event_rows",
		):
			with self.subTest(query=query), self.assertRaises(ValueError):
				store.execute_readonly_query(query)

	def test_sql_sandbox_contains_only_latest_500_rows(self) -> None:
		store = TelemetryStore()
		start = datetime(2026, 1, 1)
		for index in range(505):
			store.insert_event(
				{"timestamp": start + timedelta(minutes=index), "sequence": index}
			)

		result = store.execute_readonly_query(
			"SELECT MIN(timestamp) AS first_row, COUNT(*) AS row_count "
			"FROM recent_event_rows"
		)

		self.assertEqual(result.loc[0, "row_count"], 500)
		self.assertEqual(result.loc[0, "first_row"], start + timedelta(minutes=5))

	def test_drift_stats_detect_shift(self) -> None:
		statistics = compare_distributions(range(100), range(100, 200))

		self.assertGreater(statistics["ks_statistic"], 0.9)
		self.assertGreater(statistics["psi"], 0.2)

	def test_pdf_export_and_webhook_opt_in(self) -> None:
		pdf = create_executive_pdf("### Executive Brief\nEvidence-based finding")

		self.assertTrue(pdf.startswith(b"%PDF"))
		with patch.dict("os.environ", {}, clear=True):
			future = dispatch_webhook({"event": "test"})
		self.assertTrue(future.done())
		self.assertIn("not configured", future.result())
		self.assertTrue(RootCauseAgent._is_auth_error(Exception("401 invalid_api_key")))

	def test_configured_webhook_dispatch_is_non_blocking(self) -> None:
		worker_started = Event()
		allow_delivery = Event()

		def delayed_delivery(event: dict, target: str) -> str:
			worker_started.set()
			allow_delivery.wait(timeout=2)
			return "Webhook delivered (HTTP 200)."

		with patch("core.alerts._post_webhook", side_effect=delayed_delivery):
			future = dispatch_webhook({"event": "test"}, endpoint="https://example.invalid")
			self.assertTrue(worker_started.wait(timeout=1))
			self.assertFalse(future.done())
			allow_delivery.set()
			self.assertIn("delivered", future.result(timeout=1))


if __name__ == "__main__":
	unittest.main()
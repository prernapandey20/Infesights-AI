from datetime import datetime
import json
import re
from uuid import uuid4

import duckdb
import pandas as pd


class TelemetryStore:
	def __init__(self) -> None:
		self.connection = duckdb.connect(":memory:")
		self.connection.execute("SET enable_external_access = false")
		self.connection.execute(
			"CREATE TABLE event_rows (timestamp TIMESTAMP, row_json JSON)"
		)
		self.connection.execute(
			"""
			CREATE TABLE telemetry_stream (
				timestamp TIMESTAMP,
				feature_name VARCHAR,
				metric_value DOUBLE,
				rolling_mean DOUBLE
			)
			"""
		)
		self.connection.execute(
			"""
			CREATE TABLE incident_alerts (
				incident_id VARCHAR,
				timestamp TIMESTAMP,
				metric_name VARCHAR,
				anomalous_value DOUBLE,
				baseline_mean DOUBLE,
				confidence_score DOUBLE,
				status VARCHAR
			)
			"""
		)

	def insert_event(self, record: dict) -> None:
		self.connection.execute(
			"INSERT INTO event_rows (timestamp, row_json) VALUES (?, CAST(? AS JSON))",
			[record["timestamp"], json.dumps(record, default=str)],
		)

	def execute_readonly_query(self, query: str, limit: int = 100) -> pd.DataFrame:
		statements = duckdb.extract_statements(query)
		if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
			raise ValueError("Only one read-only SELECT statement is allowed")
		if not re.search(
			r"\bFROM\s+(?:main\.)?\"?recent_event_rows\"?(?:\s|$)",
			query,
			re.IGNORECASE,
		):
			raise ValueError("Queries must read from the recent_event_rows table")
		if limit < 1:
			raise ValueError("limit must be positive")
		recent_rows = self.connection.execute(
			"SELECT timestamp, row_json FROM event_rows "
			"ORDER BY timestamp DESC LIMIT 500"
		).fetchall()
		query_connection = duckdb.connect(":memory:")
		try:
			query_connection.execute("SET enable_external_access = false")
			query_connection.execute(
				"CREATE TABLE recent_event_rows (timestamp TIMESTAMP, row_json JSON)"
			)
			query_connection.executemany(
				"INSERT INTO recent_event_rows VALUES (?, ?)", recent_rows
			)
			return query_connection.execute(query).fetchdf().head(limit)
		finally:
			query_connection.close()

	def insert_telemetry(
		self,
		timestamp: datetime,
		feature_name: str,
		value: float,
		rolling_mean: float,
	) -> None:
		self.connection.execute(
			"""
			INSERT INTO telemetry_stream
				(timestamp, feature_name, metric_value, rolling_mean)
			VALUES (?, ?, ?, ?)
			""",
			[timestamp, feature_name, value, rolling_mean],
		)

	def get_recent_telemetry(self, limit: int = 100) -> pd.DataFrame:
		if limit < 0:
			raise ValueError("limit must be non-negative")
		return self.connection.execute(
			"""
			SELECT timestamp, feature_name, metric_value, rolling_mean
			FROM telemetry_stream
			ORDER BY timestamp DESC
			LIMIT ?
			""",
			[limit],
		).fetchdf()

	def log_incident(
		self,
		metric_name: str,
		anomalous_value: float,
		baseline_mean: float,
		confidence_score: float,
	) -> None:
		self.connection.execute(
			"""
			INSERT INTO incident_alerts
				(incident_id, timestamp, metric_name, anomalous_value,
				 baseline_mean, confidence_score, status)
			VALUES (?, ?, ?, ?, ?, ?, ?)
			""",
			[
				str(uuid4()),
				datetime.now(),
				metric_name,
				anomalous_value,
				baseline_mean,
				confidence_score,
				"OPEN",
			],
		)

	def get_open_incidents(self) -> pd.DataFrame:
		return self.connection.execute(
			"""
			SELECT incident_id, timestamp, metric_name, anomalous_value,
				   baseline_mean, confidence_score, status
			FROM incident_alerts
			WHERE status = 'OPEN'
			ORDER BY timestamp DESC
			"""
		).fetchdf()

	def get_recent_incidents(self, limit: int = 10) -> pd.DataFrame:
		if limit < 0:
			raise ValueError("limit must be non-negative")
		return self.connection.execute(
			"""
			SELECT incident_id, timestamp, metric_name, anomalous_value,
			       baseline_mean, confidence_score, status
			FROM incident_alerts
			ORDER BY timestamp DESC
			LIMIT ?
			""",
			[limit],
		).fetchdf()

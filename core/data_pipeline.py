from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
import re

import numpy as np
import pandas as pd


_IDENTIFIER_NAME_PATTERN = re.compile(
	r"(?:^|_)(?:id|uuid|key|index|postal|zip|zipcode|rowid|row)(?:$|_)"
)


def is_metric_column(column: str, series: pd.Series) -> bool:
	normalized_name = re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_")
	if _IDENTIFIER_NAME_PATTERN.search(normalized_name):
		return False

	values = series.dropna()
	if not len(values):
		return False
	unique_ratio = values.nunique() / len(values)
	if unique_ratio >= 0.98 and (
		"code" in normalized_name
		or normalized_name.endswith("number")
		or normalized_name.endswith("no")
	):
		return False
	return True


def load_dataset(data: bytes, filename: str) -> pd.DataFrame:
	if filename.lower().endswith(".csv"):
		try:
			return pd.read_csv(BytesIO(data))
		except UnicodeDecodeError:
			return pd.read_csv(BytesIO(data), encoding="ISO-8859-1")
	if filename.lower().endswith(".json"):
		payload = json.loads(data)
		if isinstance(payload, dict):
			payload = payload.get("records", payload.get("data", payload))
		if not isinstance(payload, (dict, list)):
			raise ValueError("JSON must contain an object or an array of records")
		return pd.DataFrame(payload)
	raise ValueError("Upload a CSV or JSON file")


def prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
	if frame.empty:
		raise ValueError("The dataset contains no records")

	prepared = frame.copy()
	if prepared.columns.empty:
		raise ValueError("The dataset contains no columns")
	prepared.columns = [str(column) for column in prepared.columns]

	time_column = next(
		(
			column
			for column in prepared.columns
			if str(column).strip().lower() in {"timestamp", "datetime", "date", "time"}
		),
		None,
	)
	if time_column is None:
		timestamps = pd.Series(pd.date_range(end=datetime.now(), periods=len(prepared)))
	else:
		timestamps = pd.to_datetime(prepared[time_column], errors="coerce", utc=True)
		timestamps = timestamps.dt.tz_convert(None)
		fallback = pd.Series(pd.date_range(end=datetime.now(), periods=len(prepared)))
		timestamps = timestamps.fillna(fallback)
		if time_column != "timestamp":
			prepared = prepared.drop(columns=[time_column])
	prepared["timestamp"] = timestamps.to_numpy()

	numeric_features: list[str] = []
	for column in prepared.columns:
		if (
			column == "timestamp"
			or pd.api.types.is_bool_dtype(prepared[column])
			or pd.api.types.is_datetime64_any_dtype(prepared[column])
		):
			continue
		values = pd.to_numeric(prepared[column], errors="coerce")
		values = values.replace([np.inf, -np.inf], np.nan)
		if values.notna().any() and is_metric_column(str(column), values):
			prepared[column] = values.fillna(values.median()).astype(float)
			numeric_features.append(str(column))

	if not numeric_features:
		raise ValueError("No numerical columns were found in the dataset")
	prepared["timestamp"] = pd.to_datetime(prepared["timestamp"]).dt.to_pydatetime()
	return prepared, numeric_features


def create_demo_dataset() -> pd.DataFrame:
	rng = np.random.default_rng(42)
	row_count = 240
	return pd.DataFrame(
		{
			"timestamp": pd.date_range(end=datetime.now(), periods=row_count, freq="5min"),
			"order_value": rng.normal(86, 24, row_count).clip(5),
			"delivery_delay_hours": rng.normal(4.5, 2.0, row_count).clip(0),
			"inventory_gap_units": rng.normal(7, 4, row_count).clip(0),
			"fraud_score": rng.beta(1.5, 18, row_count),
			"payment_retries": rng.poisson(0.4, row_count),
			"region": rng.choice(["north", "south", "east", "west"], row_count),
		}
	)
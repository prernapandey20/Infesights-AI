from collections import deque
from numbers import Real
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
	def __init__(self) -> None:
		self.model = IsolationForest(
			contamination=0.01,
			random_state=42,
			n_estimators=100,
		)
		self.buffer: deque[dict[str, float]] = deque(maxlen=100)
		self.feature_names: list[str] = []
		self.baseline_means: dict[str, float] = {}
		self.baseline_stds: dict[str, float] = {}
		self.deviation_threshold = 2.5
		self.alert_confidence_threshold = 98.0
		self._is_fitted = False

	def fit_initial_baseline(self, warmup_data: Iterable[dict[str, Any]]) -> None:
		records = list(warmup_data)
		if not records:
			raise ValueError("warmup_data must contain at least one record")

		self.feature_names = [
			name
			for name in records[0]
			if isinstance(records[0][name], Real)
			and not isinstance(records[0][name], bool)
		]
		if not self.feature_names:
			raise ValueError("warmup_data must contain numeric metrics")

		numeric_records = [self._extract_features(record) for record in records]
		values = np.asarray(
			[[record[name] for name in self.feature_names] for record in numeric_records],
			dtype=float,
		)
		self.baseline_means = dict(zip(self.feature_names, values.mean(axis=0)))
		self.baseline_stds = dict(
			zip(self.feature_names, np.maximum(values.std(axis=0), np.finfo(float).eps))
		)
		baseline_stds = np.maximum(values.std(axis=0), np.finfo(float).eps)
		baseline_deviations = np.max(
			np.abs((values - values.mean(axis=0)) / baseline_stds),
			axis=1,
		)
		self.deviation_threshold = float(max(2.5, np.quantile(baseline_deviations, 0.98)))
		self.model.fit(values)
		baseline_confidences = np.clip(
			(0.5 - self.model.decision_function(values)) * 100,
			0,
			100,
		)
		self.alert_confidence_threshold = float(
			np.clip(np.quantile(baseline_confidences, 0.98), 50, 99)
		)
		self.buffer.clear()
		self.buffer.extend(numeric_records[-100:])
		self._is_fitted = True

	def predict(
		self,
		record_dict: dict[str, Any],
		update_buffer: bool = True,
	) -> dict[str, Any]:
		if not self._is_fitted:
			raise RuntimeError("fit_initial_baseline must be called before predict")

		numeric_record = self._extract_features(record_dict)
		values = np.asarray(
			[[numeric_record[name] for name in self.feature_names]],
			dtype=float,
		)
		decision_score = float(self.model.decision_function(values)[0])
		confidence = float(np.clip((0.5 - decision_score) * 100, 0, 100))
		deviations = {
			name: float(abs(
				(numeric_record[name] - self.baseline_means[name])
				/ self.baseline_stds[name]
			))
			for name in self.feature_names
		}
		is_forest_anomaly = bool(
			self.model.predict(values)[0] == -1 and decision_score < -0.15
		)
		is_extreme_outlier = bool(max(deviations.values()) >= self.deviation_threshold)
		if update_buffer:
			self.buffer.append(numeric_record)

		return {
			"is_anomaly": is_forest_anomaly or is_extreme_outlier,
			"confidence": confidence,
			"alert_threshold": self.alert_confidence_threshold,
			"deviation_threshold": self.deviation_threshold,
			"deviations": deviations,
		}

	def _extract_features(self, record: dict[str, Any]) -> dict[str, float]:
		missing = [name for name in self.feature_names if name not in record]
		if missing:
			raise ValueError(f"Record is missing features: {', '.join(missing)}")
		nonnumeric = [
			name
			for name in self.feature_names
			if not isinstance(record[name], Real) or isinstance(record[name], bool)
		]
		if nonnumeric:
			raise ValueError(f"Record contains non-numeric features: {', '.join(nonnumeric)}")
		return {name: float(record[name]) for name in self.feature_names}

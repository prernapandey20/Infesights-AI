from collections import deque
from datetime import datetime
from math import isfinite
import random
from typing import Any, Iterable


class StreamProducer:
	def __init__(self, records: Iterable[dict[str, Any]], features: list[str]) -> None:
		self.records = [dict(record) for record in records]
		if not self.records:
			raise ValueError("records must contain at least one row")
		if not features:
			raise ValueError("features must contain at least one numerical column")
		self.features = features.copy()
		self.history: deque[dict[str, Any]] = deque(maxlen=100)
		self.rolling_means: dict[str, float] = {}
		self._position = 0
		self._pending_anomaly: tuple[str, float] | None = None

	def get_next_point(self) -> dict[str, Any]:
		point = self.records[self._position % len(self.records)].copy()
		self._position += 1
		point["event_timestamp"] = point.get("timestamp")
		point["timestamp"] = datetime.now()
		if self._pending_anomaly:
			feature, magnitude = self._pending_anomaly
			if feature not in self.features:
				raise ValueError(f"Unknown feature: {feature}")
			values = [float(record[feature]) for record in self.records]
			mean = sum(values) / len(values)
			variance = sum((value - mean) ** 2 for value in values) / len(values)
			point[feature] = mean + magnitude * max(variance**0.5, 1e-9)
		self.history.append(point)
		self.rolling_means = {
			feature: sum(float(item[feature]) for item in self.history) / len(self.history)
			for feature in self.features
		}
		for feature in self.features:
			if not isfinite(float(point[feature])):
				raise ValueError(f"Feature {feature} must be a finite number")
		self._pending_anomaly = None
		return point

	def inject_anomaly(self, feature: str | None = None, magnitude: float = 5.0) -> None:
		if feature is None:
			feature = random.choice(self.features)
		if feature not in self.features:
			raise ValueError(f"Unknown feature: {feature}")
		self._pending_anomaly = (feature, magnitude)

	def get_rolling_means(self) -> dict[str, float]:
		return self.rolling_means.copy()

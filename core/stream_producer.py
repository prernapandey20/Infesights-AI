from datetime import datetime
import random
from typing import Any


class StreamProducer:
	METRIC_CONFIG = {
		"cpu_utilization_pct": (55.0, 8.0),
		"memory_utilization_pct": (62.0, 6.0),
		"api_latency_ms": (180.0, 105.0),
		"error_rate_pct": (1.5, 0.4),
	}

	def __init__(self) -> None:
		self.history: list[dict[str, Any]] = []
		self.rolling_means: dict[str, float] = {
			feature: mean for feature, (mean, _) in self.METRIC_CONFIG.items()
		}
		self._pending_anomaly: tuple[str, float] | None = None

	def get_next_point(self) -> dict[str, Any]:
		point: dict[str, Any] = {"timestamp": datetime.now()}

		for feature, (mean, standard_deviation) in self.METRIC_CONFIG.items():
			value = random.gauss(mean, standard_deviation)
			if self._pending_anomaly and self._pending_anomaly[0] == feature:
				value = mean + self._pending_anomaly[1] * standard_deviation
			if feature == "api_latency_ms":
				value = max(0.0, value)
			point[feature] = value

		self.history.append(point)
		self.rolling_means = {
			feature: sum(float(item[feature]) for item in self.history) / len(self.history)
			for feature in self.METRIC_CONFIG
		}
		self._pending_anomaly = None
		return point

	def inject_anomaly(self, feature: str | None = None, magnitude: float = 5.0) -> None:
		if feature is None:
			feature = random.choice(list(self.METRIC_CONFIG))
		if feature not in self.METRIC_CONFIG:
			raise ValueError(f"Unknown feature: {feature}")
		self._pending_anomaly = (feature, magnitude)

	def get_rolling_means(self) -> dict[str, float]:
		return self.rolling_means.copy()

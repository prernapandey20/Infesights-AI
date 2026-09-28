from typing import Any

import numpy as np
from scipy.stats import ks_2samp


def compare_distributions(
	baseline: Any,
	incoming: Any,
	bins: int = 10,
) -> dict[str, float]:
	baseline_values = np.asarray(baseline, dtype=float)
	incoming_values = np.asarray(incoming, dtype=float)
	baseline_values = baseline_values[np.isfinite(baseline_values)]
	incoming_values = incoming_values[np.isfinite(incoming_values)]
	if baseline_values.size < 2 or incoming_values.size < 2:
		raise ValueError("At least two finite values are required in each window")

	ks_result = ks_2samp(baseline_values, incoming_values)
	quantiles = np.linspace(0, 1, bins + 1)[1:-1]
	cut_points = np.unique(np.quantile(baseline_values, quantiles))
	if cut_points.size == 0:
		cut_points = np.asarray([baseline_values[0]])
	bin_edges = np.concatenate(([-np.inf], cut_points, [np.inf]))
	baseline_counts = np.histogram(baseline_values, bins=bin_edges)[0]
	incoming_counts = np.histogram(incoming_values, bins=bin_edges)[0]
	baseline_share = np.maximum(baseline_counts / baseline_values.size, 1e-6)
	incoming_share = np.maximum(incoming_counts / incoming_values.size, 1e-6)
	psi = float(
		np.sum((incoming_share - baseline_share) * np.log(incoming_share / baseline_share))
	)
	return {
		"ks_statistic": float(ks_result.statistic),
		"ks_p_value": float(ks_result.pvalue),
		"psi": psi,
	}
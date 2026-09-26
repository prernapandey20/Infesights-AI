import time

from core.stream_producer import StreamProducer
from core.detector import AnomalyDetector
from core.storage import TelemetryStore


store = TelemetryStore()
producer = StreamProducer()
detector = AnomalyDetector()

# Warmup baseline
warmup = [producer.get_next_point() for _ in range(50)]
detector.fit_initial_baseline(warmup)

# Stream 5 points and inject chaos
print("Streaming test...")
for i in range(5):
    if i == 3:
        producer.inject_anomaly(feature="api_latency_ms", magnitude=8.0)
    point = producer.get_next_point()
    result = detector.predict(point)
    print(f"Tick {i}: Anomaly={result['is_anomaly']} | Conf={result['confidence']:.1f}%")
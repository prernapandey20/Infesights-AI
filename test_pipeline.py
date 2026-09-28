from core.stream_producer import StreamProducer
from core.detector import AnomalyDetector
from core.data_pipeline import create_demo_dataset, prepare_dataset


frame, features = prepare_dataset(create_demo_dataset())
records = frame.to_dict(orient="records")
producer = StreamProducer(records, features)
detector = AnomalyDetector()

detector.fit_initial_baseline(records)

print("Streaming test...")
for i in range(5):
    if i == 3:
        producer.inject_anomaly(feature=features[0], magnitude=8.0)
    point = producer.get_next_point()
    result = detector.predict(point)
    print(
        f"Tick {i}: Anomaly={result['is_anomaly']} | "
        f"Conf={result['confidence']:.1f}%"
    )
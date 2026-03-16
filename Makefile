.PHONY: install produce process build-dataset serve test docker-up docker-down clean lint

PYTHON := python
PIP    := pip

install:
	$(PIP) install -r requirements.txt

produce:
	$(PYTHON) scripts/produce_events.py --n-events 10000

produce-stream:
	$(PYTHON) scripts/produce_events.py --stream --duration 60

process:
	$(PYTHON) scripts/run_processor.py

build-dataset:
	$(PYTHON) scripts/build_training_set.py --n-events 50000 --output data/training_dataset.csv

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

test-features:
	pytest tests/test_features.py -v

test-stores:
	pytest tests/test_stores.py -v

test-pit:
	pytest tests/test_pit_join.py -v

test-consistency:
	pytest tests/test_consistency.py -v

docker-up:
	docker compose up -d
	@echo "Waiting for Kafka to be ready..."
	@sleep 20
	@echo "Services started. API at http://localhost:8000"
	@echo "Grafana at http://localhost:3000 (admin/featureflow)"
	@echo "Prometheus at http://localhost:9090"

docker-down:
	docker compose down -v

docker-logs:
	docker compose logs -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf data/offline data/training_dataset.csv
	rm -rf .pytest_cache

lint:
	python -m py_compile src/config.py src/events/schema.py src/events/generator.py \
		src/kafka/producer.py src/kafka/consumer.py \
		src/features/definitions.py src/features/transformations.py src/features/registry.py \
		src/stores/online_store.py src/stores/offline_store.py \
		src/pipeline/stream_processor.py src/pipeline/batch_processor.py \
		src/training/dataset_builder.py \
		src/serving/app.py src/serving/middleware.py \
		src/consistency/checker.py
	@echo "All files compile cleanly."

.PHONY: help setup install-prod install-dev run-cpu run-gpu stop run-local run-worker run-worker-local run-temporal-local stop-temporal-local

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  setup                - Copy .env.example to .env"
	@echo "  install-prod         - Install production dependencies"
	@echo "  install-dev          - Install development dependencies"
	@echo "  run-cpu              - Run the application in a CPU-only environment with Docker"
	@echo "  run-gpu              - Run the application in a GPU-accelerated environment with Docker"
	@echo "  stop                 - Stop the Docker containers"
	@echo "  run-local            - Run the FastAPI server locally"
	@echo "  run-worker           - Start Temporal server (if needed) and run worker"
	@echo "  run-worker-local     - Run the Temporal worker locally"
	@echo "  run-temporal-local   - Start the Temporal server locally (if not running)"
	@echo "  stop-temporal-local  - Stop the local Temporal server"

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env file created. Please fill in your details."; \
	else \
		echo ".env file already exists."; \
	fi

install-prod:
	pip install -r requirements/prod.txt

install-dev:
	pip install -r requirements/dev.txt

run-cpu:
	docker-compose -f docker/docker-compose.cpu.yml up --build

run-gpu:
	docker-compose -f docker/docker-compose.gpu.yml -f docker/docker-compose.temporal.yml up --build

stop:
	docker-compose -f docker/docker-compose.cpu.yml down
	docker-compose -f docker/docker-compose.gpu.yml -f docker/docker-compose.temporal.yml down

run-local:
	uvicorn app.main:app --reload --log-config app/uvicorn_log_conf.yaml --log-level info

run-worker: run-temporal-local run-worker-local

run-worker-local:
	python -m app.temporal_worker

run-temporal-local:
	@echo "Starting local Temporal server..."
	@if ! lsof -Pi :7233 -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "Temporal server not running, starting it..."; \
		temporal server start-dev & \
		echo "Temporal server started in background"; \
		sleep 5; \
	else \
		echo "Temporal server already running on port 7233"; \
	fi

stop-temporal-local:
	@echo "Stopping local Temporal server..."
	@if lsof -Pi :7233 -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "Found Temporal server running on port 7233, stopping it..."; \
		pkill -f "temporal server start-dev" || true; \
		sleep 2; \
		echo "Temporal server stopped"; \
	else \
		echo "Temporal server is not running on port 7233"; \
	fi

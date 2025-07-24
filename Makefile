.PHONY: help setup install-prod install-dev run-cpu run-gpu stop run-local run-worker-local run-temporal-local stop-temporal-local

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
	@echo "  run-worker-local     - Run the Temporal worker locally"
	@echo "  run-temporal-local   - Run the Temporal server locally"
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

run-worker-local:
	python -m app.temporal_worker

run-temporal-local:
	@echo "Starting local Temporal server..."
	@temporal server start-dev > /tmp/temporal.log 2>&1 & echo $! > .temporal.pid
	@echo "Temporal server started with PID $(cat .temporal.pid)"

stop-temporal-local:
	@if [ -f .temporal.pid ]; then \
		echo "Stopping local Temporal server..."; \
		kill $(cat .temporal.pid) && rm .temporal.pid; \
		echo "Temporal server stopped."; \
	else \
		echo "Temporal server not running or .temporal.pid not found."; \
	fi

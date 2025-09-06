.PHONY: run-cpu run-gpu stop run run-local run-worker run-worker-local run-temporal-local stop-temporal-local

install-prod:
	uv sync --no-dev

install-prod-gpu:
	uv sync --no-dev --extra gpu

install-dev:
	uv sync

install-dev-gpu:
	uv sync --extra gpu

run: run-worker run-local

run-local:
	uvicorn app.main:app --reload --log-config app/uvicorn_log_conf.yaml --log-level info

run-worker: run-temporal-local run-worker-local

run-worker-local:
	python -m app.temporal_worker &

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

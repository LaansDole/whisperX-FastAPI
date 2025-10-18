.PHONY: help install-prod install-prod-gpu install-dev install-dev-gpu \
	dev server worker start-temporal \
	start-temporal stop-temporal

# Default target - show help
help:
	@echo "Available targets:"
	@echo "  install-prod         - Install production dependencies (CPU only)"
	@echo "  install-prod-gpu     - Install production dependencies with GPU support"
	@echo "  install-dev          - Install development dependencies (CPU only)"
	@echo "  install-dev-gpu      - Install development dependencies with GPU support"
	@echo "  dev                  - Start worker + FastAPI server (full app)"
	@echo "  server            - Start FastAPI server only"
	@echo "  worker           - Start Temporal server + worker"
	@echo "  start-temporal     - Start Temporal worker only"
	@echo "  start-temporal   - Start local Temporal server"
	@echo "  stop-temporal  - Stop local Temporal server"

# ============================================================================
# Installation targets
# ============================================================================

install-prod:
	uv sync --no-dev

install-prod-gpu:
	uv sync --no-dev --extra gpu

install-dev:
	uv sync

install-dev-gpu:
	uv sync --extra gpu

# ============================================================================
# Run targets
# ============================================================================

# Start full application (Temporal worker + FastAPI server)
dev:
	@echo "Starting full application..."
	$(MAKE) worker
	@echo "Waiting for worker to initialize..."
	uv run python scripts/wait_for_worker.py
	$(MAKE) server
	@echo "✓ Full application started"

# Start FastAPI server only
server:
	uv run python -m start_server
	@echo "✓ FastAPI server started"

# Start Temporal server + worker
worker: stop-temporal start-temporal
	uv run python -m app.temporal_worker &
	@echo "✓ Temporal worker started"

# ============================================================================
# Temporal server management
# ============================================================================

# Start local Temporal server if not already devning
start-temporal:
	@echo "Checking Temporal server status..."
	@if ! lsof -Pi :7233 -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "Starting Temporal server..."; \
		temporal server start-dev & \
		sleep 5; \
		echo "✓ Temporal server started on port 7233"; \
	else \
		echo "✓ Temporal server already devning on port 7233"; \
	fi

# Stop local Temporal server
stop-temporal:
	@echo "Stopping Temporal server..."
	@if lsof -Pi :7233 -sTCP:LISTEN -t >/dev/null 2>&1; then \
		pkill -f "temporal_worker" || true; \
		sleep 2; \
		echo "✓ Temporal server stopped"; \
	else \
		echo "✓ Temporal server is not devning"; \
	fi

.PHONY: help install-prod install-prod-gpu install-dev install-dev-gpu \
	dev server worker streamlit start-temporal lint format \
	stop-temporal stop test-api temporal-fresh check-activities \
	test test-unit test-integration test-medical test-coverage test-all test-quick \
	docker-build docker-up docker-start docker-restart docker-down down

# Default target - show help
help:
	@echo "Available targets:"
	@echo "  install-prod         	- Install production dependencies (CPU only)"
	@echo "  install-prod-gpu     	- Install production dependencies with GPU support"
	@echo "  install-dev          	- Install development dependencies (CPU only)"
	@echo "  install-dev-gpu      	- Install development dependencies with GPU support"
	@echo "  dev                  	- Start worker + FastAPI server + Streamlit UI"
	@echo "  server            		- Start FastAPI server only"
	@echo "  worker           		- Start Temporal server + worker"
	@echo "  streamlit        		- Start Streamlit UI only"
	@echo "  start-temporal     	- Start local Temporal server"
	@echo "  stop-temporal      	- Stop local Temporal server"
	@echo "  stop              		- Stop all running processes (pkill)"
	@echo "  temporal-fresh     	- Stop all workers, clean Temporal data, and start fresh"
	@echo "  check-activities  	- Check running Temporal activities via CLI"
	@echo ""
	@echo "Docker targets:"
	@echo "  docker-build      	- Build Docker images only"
	@echo "  docker-up         	- Build and start all services (with build)"
	@echo "  docker-start      	- Start services without rebuild (faster)"
	@echo "  docker-restart    	- Restart running services"
	@echo "  docker-down       	- Stop and remove all containers"
	@echo ""
	@echo "Code quality targets:"
	@echo "  lint              	- Run all linting checks (ruff, yamllint, etc.)"
	@echo "  format            	- Format code with ruff"
	@echo ""
	@echo "Testing targets:"
	@echo "  test              	- Run all tests (unit + integration)"
	@echo "  test-unit         	- Run unit tests only"
	@echo "  test-integration  	- Run integration tests only"
	@echo "  test-medical      	- Run medical RAG tests only"
	@echo "  test-quick        	- Run tests excluding slow ones"
	@echo "  test-coverage     	- Run tests with coverage report"
	@echo "  test-all          	- Run comprehensive test suite"
	@echo ""
	@echo "Environment Variables:"
	@echo "  TEMPORAL_DB_PATH  	- Path for Temporal database (default: ./temporal_data/temporal.db)"

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
# Code quality targets
# ============================================================================

# Run all linting checks based on .pre-commit-config.yaml
lint:
	@echo "Running linting checks..."
	@echo "✓ Running ruff linter (required)..."
	uv run ruff check app/ tests/ streamlit_app/ --config pyproject.toml
	@echo ""
	@echo "✓ Running optional linters..."
	@echo "  - yamllint (YAML files)..."
	@uv run yamllint -d "{extends: relaxed, rules: {line-length: disable}}" -s . 2>/dev/null || echo "    ⚠ yamllint not installed (optional)"
	@echo "  - pydocstyle (docstrings)..."
	@uv run pydocstyle app/ 2>/dev/null || echo "    ⚠ pydocstyle not installed (optional)"
	@echo "  - codespell (spelling)..."
	@uv run codespell app/ tests/ 2>/dev/null || echo "    ⚠ codespell not installed (optional)"
	@echo ""
	@echo "✅ All linting checks completed!"

# Format code with ruff
format:
	@echo "Formatting code with ruff..."
	uv run ruff check app/ tests/ streamlit_app/ --fix --unsafe-fixes --config pyproject.toml
	uv run ruff format app/ tests/ streamlit_app/ --config pyproject.toml
	@echo "Code formatting completed"

# ============================================================================
# Run targets
# ============================================================================

# Start full application (Temporal worker + FastAPI server + Streamlit UI)
dev:
	@echo "Starting full application..."
	$(MAKE) worker
	@echo "Waiting for worker to initialize..."
	uv run python scripts/wait_for_worker.py
	$(MAKE) server
	@echo "Waiting for server to start..."
	uv run python scripts/wait_for_server.py
	$(MAKE) streamlit
	@echo "Full application started"
	@$(MAKE) list-servers

# Start Streamlit UI only
streamlit:
	uv run streamlit run streamlit_app/app.py --server.port 8501 &
	@echo "Streamlit UI started at http://localhost:8501"

# Start FastAPI server only
server:
	uv run python -m start_server &
	@echo "FastAPI server started at http://localhost:8000"

# Start Temporal server + worker
worker: stop start-temporal
	uv run python -m app.temporal.worker &
	@echo "Temporal worker started"

# ============================================================================
# Temporal server management
# ============================================================================

# Define the Temporal database location for controlled persistence
TEMPORAL_DB_PATH ?= ./temporal_data/temporal.db

# Start local Temporal server with controlled database location
start-temporal:
	@echo "Checking Temporal server status..."
	@if ! lsof -Pi :7233 -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "Starting Temporal server with controlled database at $(TEMPORAL_DB_PATH)..."; \
		mkdir -p $$(dirname "$(TEMPORAL_DB_PATH)") 2>/dev/null || true; \
		temporal server start-dev --db-filename "$(TEMPORAL_DB_PATH)" & \
		sleep 5; \
		echo "Temporal server started on port 7233"; \
		echo "Database location: $(TEMPORAL_DB_PATH)"; \
	else \
		echo "Temporal server already running on port 7233"; \
		echo "Database location: $(TEMPORAL_DB_PATH)"; \
	fi

# ============================================================================
# Process management
# ============================================================================

list-servers:
	@echo "List of running servers:"
	@echo "  FastAPI:   http://localhost:8000"
	@echo "  Admin:     http://localhost:8000/admin"
	@echo "  Streamlit: http://localhost:8501"
	@echo "  Temporal:  http://localhost:8233"

# Stop all running processes (FastAPI, Temporal, worker, etc.)
stop:
	@echo "Stopping all running processes..."
	@echo "Stopping FastAPI server processes..."
	@pkill -f "uvicorn.*app.main:app" || true
	@pkill -f "uvicorn.*main:app" || true
	@pkill -f "start_server" || true
	@echo "Stopping Streamlit processes..."
	@pkill -f "streamlit run" || true
	@echo "Stopping Temporal processes..."
	@pkill -f "temporal server" || true
	@pkill -f "temporal_server" || true
	@pkill -f "temporal.worker" || true
	@pkill -f "app.temporal.worker" || true
	@echo "Stopping Python processes..."
	@pkill -f "python.*whisperx" || true
	@pkill -f "python.*app" || true
	@echo "Stopping any remaining uvicorn processes..."
	@pkill -f "uvicorn" || true
	@echo "All processes stopped"
	@echo ""
	@echo "Cleanup complete. All related processes have been terminated."

# ============================================================================
# Testing targets
# ============================================================================

# Clean Temporal data and start fresh
temporal-fresh:
	@echo "🧹 CLEANING TEMPORAL DATA FOR FRESH START"
	@echo "============================================"
	@echo ""
	@echo "1️⃣  Stopping all processes (workers, server, etc.)..."
	$(MAKE) stop
	@echo "   Ensuring Temporal server is completely stopped..."
	@pkill -9 -f "temporal server" || true
	@echo " ✓ All processes stopped (including workers)"
	@echo ""
	@echo "2️⃣  Cleaning Temporal data directories..."
	@echo "Removing controlled Temporal database: $(TEMPORAL_DB_PATH)..."
	@if [ -f "$(TEMPORAL_DB_PATH)" ]; then \
		echo "Removing database file: $(TEMPORAL_DB_PATH)"; \
		rm -f "$(TEMPORAL_DB_PATH)"; \
		echo " ✓ Temporal database removed"; \
	else \
		echo "Temporal database not found at $(TEMPORAL_DB_PATH)"; \
	fi
	@if [ -d "$$(dirname "$(TEMPORAL_DB_PATH)")" ]; then \
		echo "Removing database directory: $$(dirname "$(TEMPORAL_DB_PATH)")"; \
		rmdir $$(dirname "$(TEMPORAL_DB_PATH)") 2>/dev/null || true; \
	fi
	@if [ -d "/tmp/temporal" ]; then \
		echo "Removing /tmp/temporal directory..."; \
		rm -rf /tmp/temporal; \
		echo " ✓ /tmp/temporal removed"; \
	else \
		echo "/tmp/temporal directory not found"; \
	fi
	@if [ -d "~/.temporal" ]; then \
		echo "Removing ~/.temporal directory..."; \
		rm -rf ~/.temporal; \
		echo " ✓ ~/.temporal removed"; \
	else \
		echo "~/.temporal directory not found"; \
	fi
	@echo ""
	@echo "3️⃣  Cleaning Temporal databases..."
	@echo "Searching for Temporal database files..."
	@find /tmp -name "*.db" -name "*temporal*" -o -name "*.temporal_*" 2>/dev/null | while read db_file; do \
		echo "Removing database file: $$db_file"; \
		rm -f "$$db_file" 2>/dev/null || true; \
	done
	@if [ -f "/var/db/temporal.db" ]; then \
		echo "Removing main database: /var/db/temporal.db"; \
		rm -f /var/db/temporal.db 2>/dev/null || true; \
	fi
	echo " ✓ Temporal database files cleaned"
	@echo ""
	@echo "4️⃣  Cleaning Temporal logs..."
	@if [ -d "./temporal_logs" ]; then \
		echo "Removing ./temporal_logs directory..."; \
		rm -rf ./temporal_logs; \
		echo " ✓ Temporal logs removed"; \
	else \
		echo "Temporal logs directory not found"; \
	fi
	@echo ""
	@echo "5️⃣  Clearing Docker Temporal images (if present)..."
	@echo "Checking for Temporal Docker images..."
	@docker images | grep "temporalio" 2>/dev/null | while read line; do \
		echo "Found Temporal image: $$line"; \
		image_id=$$(echo $$line | awk '{print $$3}'); \
		if [ -n "$$image_id" ]; then \
			echo "Removing image: $$image_id"; \
			docker rmi "$$image_id" 2>/dev/null || true; \
		fi; \
	done || echo "Docker not available or no Temporal images found"
	@echo ""
	@echo "Cleaning Temporal CLI configuration..."
	@echo "Clearing Temporal CLI namespace data..."
	@if temporal --version > /dev/null 2>&1; then \
		echo "Clearing Temporal CLI namespace data..." && \
		temporal operator namespace describe default > /dev/null 2>&1 && temporal workflow list --namespace default --limit 1 | grep -q . && \
		temporal workflow list --namespace default | awk 'NR>1 {print $$2}' | xargs -I {} temporal workflow delete --namespace default --workflow-id {} 2>/dev/null || true && \
		echo " ✓ Temporal CLI namespace cleared" || echo "Could not clear Temporal CLI namespace"; \
	else \
		echo "Temporal CLI not available"; \
	fi
	@echo ""
	@echo "7️⃣  Starting fresh Temporal server..."
	$(MAKE) start-temporal
	@echo " ✓ Fresh Temporal server started"
	@echo ""
	@echo "8️⃣  Waiting for Temporal server to initialize..."
	sleep 8
	@echo " ✓ Temporal server should be ready"
	@echo ""
	@echo "9️⃣  Opening Temporal UI in browser..."
	@echo "Opening http://localhost:8233 (Temporal UI) in browser..."
	@open http://localhost:8233 2>/dev/null || echo "Temporal UI available at: http://localhost:8233"
	@echo ""
	@echo "============================================"
	@echo "TEMPORAL FRESH START COMPLETED!"
	@echo " ✓ All old workflows cleared from Temporal UI"
	@echo " ✓ All databases and logs cleaned"
	@echo " ✓ Fresh Temporal server started with database at $(TEMPORAL_DB_PATH)"
	@echo " ✓ Clean Temporal CLI namespace"

	@echo "============================================"

# ============================================================================
# Testing targets
# ============================================================================

# Run all tests (unit + integration)
test:
	@echo "Running all tests..."
	uv run pytest tests/ -v --tb=short

unit-test:
	@echo "========================================" 
	@echo "Running Unit Tests with Coverage Report"
	@echo "========================================"
	@uv run pytest tests/unit/ -v --tb=short --cov=app --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "📊 Coverage report generated in htmlcov/index.html"
	@echo "ℹ️  Open with: open htmlcov/index.html"

integration-test:
	@echo "========================================"
	@echo "Running Integration Tests"
	@echo "========================================"
	@echo "Make sure server is running: make dev"
	uv run pytest tests/integration/ -v --tb=short -s && \
	echo "Integration tests passed"

# Run unit tests only (exclude integration and slow tests)
test-unit:
	@echo "Running unit tests only..."
	uv run pytest tests/ -v -m "not integration and not slow"
	@echo "Unit tests completed"

# Run integration tests only
test-integration:
	@echo "Running integration tests only..."
	uv run pytest tests/ -v -m integration
	@echo "Integration tests completed"

# Run medical RAG tests only
test-medical:
	@echo "Running medical RAG tests..."
	uv run pytest tests/ -v -m medical
	@echo "Medical tests completed"

# Run tests excluding slow ones
test-quick:
	@echo "Running quick tests (excluding slow ones)..."
	uv run pytest tests/ -v -m "not slow"
	@echo "Quick tests completed"

# Run tests with coverage report
test-coverage:
	@echo "Running tests with coverage report..."
	uv run pytest tests/ --cov=app --cov-report=term-missing --cov-report=html:htmlcov -v
	@echo "Coverage report generated"
	@echo "HTML coverage report available at: htmlcov/index.html"

# Run comprehensive test suite with all options
test-all:
	@echo "Running comprehensive test suite..."
	@echo "This includes all tests with coverage and detailed output..."
	uv run pytest tests/ --cov=app --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml -v --tb=short
	@echo "Comprehensive test suite completed"
	@echo "HTML coverage report available at: htmlcov/index.html"
	@echo "XML coverage report available at: coverage.xml"

# ============================================================================
# Activity monitoring
# ============================================================================

# Check running Temporal activities via CLI
check-activities:
	@echo "CHECKING TEMPORAL ACTIVITIES"
	@echo "================================="
	@echo ""
	@echo "1. Checking Temporal CLI connection..."
	@if ! temporal --version > /dev/null 2>&1; then \
		echo "Temporal CLI not available"; \
		exit 1; \
	fi
	@echo "Temporal CLI available"
	@echo ""
	@echo "2. Checking workflows..."
	@if temporal workflow list --namespace default > /dev/null 2>&1; then \
		WORKFLOW_COUNT=$$(temporal workflow list --namespace default 2>/dev/null | grep -c "running\|completed\|failed" || echo "0"); \
		if [ "$$WORKFLOW_COUNT" -eq 0 ]; then \
			echo "No workflows found in default namespace"; \
		else \
			echo "Found $$WORKFLOW_COUNT workflow(s):"; \
			echo ""; \
			temporal workflow list --namespace default --output table || echo "   (Could not format as table)"; \
		fi; \
	else \
		echo "Could not connect to Temporal server"; \
		echo "   Make sure Temporal server is running: make start-temporal"; \
	fi
	@echo ""
	@echo "3. Checking task queues..."
	@if temporal task-queue list --namespace default > /dev/null 2>&1; then \
		echo "Task queues available"; \
		temporal task-queue list --namespace default 2>/dev/null | head -5 || echo "   (Could not list task queues)"; \
	else \
		echo "Could not list task queues"; \
	fi
	@echo ""
	@echo "================================="
	@echo "ACTIVITY CHECK COMPLETED"
	@echo "================================="
# ============================================================================
# Docker targets
# ============================================================================

build:
	docker-compose up --build -d
	@echo "Docker images built"
	@$(MAKE) list-servers

up:
	docker-compose up -d
	@echo "Docker services started"
	@$(MAKE) list-servers

restart:
	docker-compose restart
	@echo "Docker services restarted"
	@$(MAKE) list-servers

down:
	docker-compose down
	@echo "Docker services stopped"

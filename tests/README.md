# Test Directory Structure

This directory contains all tests for the WhisperX-FastAPI project, organized by test type.

## Directory Organization

```
tests/
├── __init__.py                 # Package init
├── conftest.py                 # Shared pytest fixtures
├── pytest.ini                  # Pytest configuration
│
├── unit/                       # Unit tests (isolated component testing)
│
├── integration/                # Integration tests (multiple components, requires running server)
│
├── services/                   # Service-specific tests (service layer testing, requires running server)
│
└── scripts/                    # Utility scripts (not actual tests)
```

## Test Categories

### Unit Tests (`tests/unit/`)
Self-contained tests for individual components and functions:
- No external dependencies required
- Fast execution
- Test single units of functionality
- Examples: Configuration, error handlers, utility functions

**Run:** `uv run python -m pytest tests/unit/`

### Integration Tests (`tests/integration/`)
Tests that verify multiple components working together:
- **Require running server** on localhost:8000
- Test API endpoints end-to-end
- May require external services (LM Studio, Temporal)
- Examples: REST API endpoints, workflow integration

**Run:** `uv run python -m pytest tests/integration/`

### Service Tests (`tests/services/`)
Comprehensive tests for service layer components:
- Test business logic and data transformation
- Include both unit and integration characteristics
- Use real or realistic test data
- Examples: WhisperX parser, speaker identifier, medical NLP services

**Run:** `uv run python -m pytest tests/services/`

### Utility Scripts (`tests/scripts/`)
Helper scripts for testing and validation (not pytest tests):
- Analysis and reporting tools
- Data processing utilities
- Run directly as Python scripts

**Run:** `uv run python tests/scripts/<script_name>.py`

## Running Tests

**All tests:**
```bash
uv run python -m pytest tests/
```

**Specific category:**
```bash
uv run python -m pytest tests/unit/
uv run python -m pytest tests/integration/
uv run python -m pytest tests/services/
```

**With coverage:**
```bash
uv run python -m pytest tests/ --cov=app --cov-report=html
```

**Integration tests only (requires server):**
```bash
# Start server first
make dev

# In another terminal
uv run python -m pytest tests/integration/ -v
```

## Test Markers

Tests use pytest markers for categorization:
- `@pytest.mark.integration` - Integration tests requiring live server
- `@pytest.mark.medical` - Medical RAG specific tests
- `@pytest.mark.slow` - Long-running tests

**Run specific markers:**
```bash
uv run python -m pytest -m integration
uv run python -m pytest -m "not slow"
```

## Writing New Tests

1. **Unit tests** → `tests/unit/test_<module>_<feature>.py`
   - Test individual functions/classes
   - Mock external dependencies
   - Fast and isolated

2. **Integration tests** → `tests/integration/test_<feature>_endpoints.py`
   - Test complete workflows
   - Use real server endpoints
   - Mark with `@pytest.mark.integration`

3. **Service tests** → `tests/services/test_<service_name>.py`
   - Test service layer logic
   - Use realistic data
   - May combine unit and integration approaches

4. **Utility scripts** → `tests/scripts/<descriptive_name>.py`
   - Add `if __name__ == "__main__":` block
   - Include usage documentation
   - Not run by pytest

## Fixtures

Shared fixtures are defined in:
- `conftest.py` - Project-wide fixtures
- Service-specific `conftest.py` files in subdirectories

## Configuration

Test configuration in `pytest.ini`:
- Test discovery patterns
- Logging settings
- Plugin configuration
- Coverage settings

# WhisperX FastAPI Project Rules

You are working on a professional speech-to-text API service built with FastAPI and WhisperX. This project provides transcription, alignment, diarization, and speaker identification services for audio and video files.

## Project Overview

This is a production-ready FastAPI application that:
- Provides speech-to-text services using OpenAI's Whisper models
- Supports audio alignment and speaker diarization
- Offers both file upload and URL-based processing
- Uses SQLAlchemy for task management and result storage
- Supports both CPU and GPU inference
- Includes comprehensive health checks and monitoring

## Core Development Principles

### Code Quality Standards

- **Python Version**: Use Python 3.8+ features and syntax
- **Type Hints**: Always include comprehensive type hints for all functions, parameters, and return values
- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Code Style**: Follow PEP 8 with Black formatting (line length: 88 characters)
- **Import Organization**: Use isort for import organization (group stdlib, third-party, local)

### FastAPI Best Practices

- **Router Organization**: Keep routers in separate files under `app/routers/`
- **Dependency Injection**: Use FastAPI's dependency injection system for database sessions, configuration, etc.
- **Pydantic Models**: Define clear request/response models in `app/schemas.py`
- **Error Handling**: Use HTTPException with appropriate status codes and detailed messages
- **Background Tasks**: Use FastAPI's BackgroundTasks for long-running operations
- **Validation**: Leverage Pydantic for request validation and serialization

### Database and ORM

- **SQLAlchemy**: Use SQLAlchemy 2.0+ async syntax where possible
- **Models**: Keep database models in `app/models.py`
- **Migrations**: Use Alembic for database migrations when needed
- **Session Management**: Always use proper session handling with dependency injection
- **Connection Pooling**: Configure appropriate connection pooling for production

### Audio Processing Guidelines

- **File Validation**: Always validate audio/video file formats before processing
- **Supported Formats**: 
  - Audio: `.oga`, `.m4a`, `.aac`, `.wav`, `.amr`, `.wma`, `.awb`, `.mp3`, `.ogg`
  - Video: `.wmv`, `.mkv`, `.avi`, `.mov`, `.mp4`
- **Model Management**: Cache whisper models to avoid repeated downloads
- **Resource Management**: Properly handle GPU/CPU resources and memory cleanup
- **Error Recovery**: Implement robust error handling for audio processing failures

### Environment and Configuration

- **Environment Variables**: Use `.env` files for configuration, never hardcode secrets
- **Required Variables**:
  - `HF_TOKEN`: Hugging Face token for model access
  - `WHISPER_MODEL`: Default model size (tiny, base, small, medium, large, etc.)
  - `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
  - `DB_URL`: Database connection string
  - `DEVICE`: Processing device (cuda/cpu)
  - `COMPUTE_TYPE`: Computation type (float16, float32, int8)

### Logging and Monitoring

- **Structured Logging**: Use the configured logging system (`app/logger.py`)
- **Health Checks**: Maintain the three health check endpoints:
  - `/health`: Basic service status
  - `/health/live`: Liveness probe with timestamp
  - `/health/ready`: Readiness probe with database connectivity check
- **Performance Monitoring**: Log processing times and resource usage
- **Error Tracking**: Comprehensive error logging with context

### Testing Standards

- **Test Organization**: Keep tests in `tests/` directory
- **Test Files**: Use `test_*.py` naming convention
- **Coverage**: Aim for high test coverage, especially for critical audio processing paths
- **Test Data**: Use the provided test files in `tests/test_files/` for consistent testing
- **Async Testing**: Use pytest-asyncio for testing async endpoints

### Security Considerations

- **Input Validation**: Validate all file uploads and URL inputs
- **File Size Limits**: Implement reasonable file size limits for uploads
- **Rate Limiting**: Consider implementing rate limiting for production use
- **Sensitive Data**: Never log or expose API keys, tokens, or sensitive configuration
- **CORS**: Configure CORS appropriately for your deployment environment

## File Organization Rules

### Directory Structure
```
app/
├── __init__.py
├── main.py              # FastAPI application entry point
├── config.py            # Configuration management
├── models.py            # SQLAlchemy database models
├── schemas.py           # Pydantic request/response models
├── services.py          # Business logic layer
├── tasks.py             # Background task definitions
├── db.py                # Database connection and session management
├── logger.py            # Logging configuration
├── whisperx_services.py # WhisperX integration layer
├── routers/             # FastAPI route handlers
│   ├── __init__.py
│   ├── stt.py          # Speech-to-text endpoints
│   ├── stt_services.py # Individual service endpoints
│   └── task.py         # Task management endpoints
└── docs/               # API documentation and schemas
```

### File Naming Conventions

- Use snake_case for Python files and directories
- Use descriptive names that clearly indicate the file's purpose
- Group related functionality in appropriately named modules

## API Design Guidelines

### Endpoint Design

- **Consistent Naming**: Use clear, RESTful endpoint names
- **HTTP Methods**: Use appropriate HTTP methods (GET, POST, PUT, DELETE)
- **Response Format**: Maintain consistent JSON response structure
- **Status Codes**: Use appropriate HTTP status codes
- **Error Responses**: Provide clear, actionable error messages

### Request/Response Models

- **Validation**: Use Pydantic models for all request/response validation
- **Documentation**: Include field descriptions and examples in Pydantic models
- **Backwards Compatibility**: Consider API versioning for breaking changes

### Background Processing

- **Task IDs**: Generate unique identifiers for all processing tasks
- **Status Tracking**: Implement comprehensive task status tracking
- **Result Storage**: Store results with appropriate retention policies
- **Progress Updates**: Provide progress information where possible

## Development Workflow

### Code Changes

- **Small Commits**: Make focused, atomic commits with clear messages
- **Testing**: Test all changes locally before committing
- **Documentation**: Update documentation for any API changes
- **Dependencies**: Use appropriate requirement files (`requirements/prod.txt`, `requirements/dev.txt`)

### Docker Development

- **Multi-stage Builds**: Use the existing Dockerfile structure
- **Environment Variables**: Ensure proper environment variable handling
- **GPU Support**: Maintain GPU support for Docker deployments
- **Volume Management**: Use appropriate volumes for model caching

### Performance Considerations

- **Async Operations**: Use async/await for I/O operations
- **Resource Management**: Properly manage memory and GPU resources
- **Caching**: Implement appropriate caching strategies
- **Concurrency**: Handle concurrent requests efficiently

## Specific Implementation Guidelines

### When Adding New Features

1. **Plan First**: Design the feature in `app/docs/` if complex
2. **Schema Definition**: Define Pydantic models in `schemas.py`
3. **Database Changes**: Update models in `models.py` if needed
4. **Service Layer**: Implement business logic in `services.py`
5. **Router Implementation**: Add endpoints in appropriate router files
6. **Testing**: Write comprehensive tests
7. **Documentation**: Update OpenAPI documentation

### When Modifying Audio Processing

- **Test with Multiple Formats**: Use various audio/video file types
- **Error Handling**: Implement robust error handling and cleanup
- **Resource Cleanup**: Ensure proper cleanup of temporary files and resources
- **Performance Testing**: Test with different model sizes and configurations

### When Working with Database

- **Session Management**: Always use proper session handling
- **Error Handling**: Handle database connection errors gracefully
- **Migration Strategy**: Plan database schema changes carefully
- **Data Validation**: Validate data before database operations

## Common Tasks and Patterns

### Adding New Endpoints

1. Define request/response schemas in `schemas.py`
2. Add business logic to `services.py`
3. Create router function in appropriate router file
4. Add error handling and validation
5. Write tests for the new endpoint
6. Update API documentation

### Adding New Audio Processing Features

1. Research WhisperX capabilities and requirements
2. Add utility functions to `whisperx_services.py`
3. Update service layer to use new functionality
4. Add appropriate error handling
5. Test with various audio formats and configurations
6. Update documentation and examples

### Debugging Issues

1. Check logs for detailed error information
2. Verify environment variable configuration
3. Test with minimal examples using provided test files
4. Check database connectivity and status
5. Verify model availability and download status
6. Test GPU/CPU configuration if performance-related

## Error Handling Patterns

- **Validation Errors**: Return 422 with detailed field-level errors
- **Processing Errors**: Return 500 with safe error messages (log details internally)
- **Not Found**: Return 404 for missing resources
- **Authentication/Authorization**: Return 401/403 as appropriate
- **Rate Limiting**: Return 429 with retry information

## Questions to Ask Before Making Changes

1. **Impact Assessment**: Will this change affect existing API consumers?
2. **Performance**: How will this change affect processing performance?
3. **Resource Usage**: Will this increase memory or GPU usage significantly?
4. **Error Handling**: What edge cases need to be considered?
5. **Testing**: What test cases are needed to verify the change?
6. **Documentation**: What documentation needs to be updated?

Remember: This is a production API service. Prioritize reliability, performance, and maintainability in all code changes. When in doubt, choose the simpler, more robust solution.

## Virtual Environment Management

### CRITICAL: Always Activate Virtual Environment

Before performing ANY development tasks, testing, or running commands, you MUST activate the virtual environment:

```bash
source venv/bin/activate
```

This ensures:
- Correct Python interpreter and package versions are used
- Dependencies are properly isolated from system packages
- PyTorch and WhisperX are available with correct CUDA support
- All project-specific packages are accessible

### Environment Setup Checklist

When starting any development session:

1. **Activate Virtual Environment**: `source venv/bin/activate`
2. **Verify Python Version**: Ensure Python 3.8+ is active
3. **Check Key Dependencies**: Verify PyTorch, WhisperX, and FastAPI are available
4. **Load Environment Variables**: Ensure `.env` file is present and loaded
5. **Test Database Connection**: Verify `DB_URL` is accessible

### Commands That Require Virtual Environment

- Installing dependencies (`pip install -r requirements/dev.txt`)
- Running the application (`uvicorn app.main:app --reload`)
- Running tests (`pytest tests/`)
- Using Python scripts or imports
- Any development or debugging tasks

### Troubleshooting Environment Issues

If you encounter import errors or package not found issues:
1. Verify virtual environment is activated (`which python` should point to venv)
2. Reinstall dependencies if needed
3. Check that PyTorch installation matches your CUDA version
4. Ensure all environment variables are properly set
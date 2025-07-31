# Temporal Retry Policies Documentation

## Overview

This document describes the retry policies implemented for the WhisperX Temporal workflows and activities.

## Retry Policy Types

### 1. Model Loading Retry Policy (`get_model_loading_retry_policy()`)
- **Used for**: Transcription and Diarization activities (heavy model loading)
- **Initial Interval**: 10 seconds
- **Backoff Coefficient**: 1.5
- **Maximum Interval**: 10 minutes
- **Maximum Attempts**: 5

### 2. Default Retry Policy (`get_default_retry_policy()`)
- **Used for**: Alignment and Speaker Assignment activities
- **Initial Interval**: 5 seconds (configurable)
- **Backoff Coefficient**: 2.0 (configurable)
- **Maximum Interval**: 5 minutes (configurable)
- **Maximum Attempts**: 3 (configurable)

### 3. GPU Memory Retry Policy (`get_gpu_memory_retry_policy()`)
- **Used for**: GPU memory related failures
- **Initial Interval**: 15 seconds
- **Backoff Coefficient**: 1.2
- **Maximum Interval**: 2 minutes
- **Maximum Attempts**: 3

## Error Classification

### Non-Retryable Errors
- **Authentication Errors**: Invalid HF_TOKEN, 401 errors
- **Configuration Errors**: Missing required configuration, invalid model names
- **Licensing Errors**: Terms of service not accepted

### Retryable Errors
- **Network Errors**: Download failures, connection timeouts
- **GPU Memory Errors**: CUDA out of memory, GPU busy
- **Temporary Errors**: Model loading failures, temporary service unavailability

## Workflow-Level Retry Policies

Each activity in the workflow has specific retry policies:

### Transcription Activity
- **Timeout**: 30 minutes (configurable via `TRANSCRIPTION_TIMEOUT`)
- **Retry Policy**: Extended model loading policy
- **Retries**: 3 attempts with exponential backoff

### Alignment Activity
- **Timeout**: 10 minutes (configurable via `ALIGNMENT_TIMEOUT`)
- **Retry Policy**: Default policy with shorter intervals
- **Retries**: 2 attempts with moderate backoff

### Diarization Activity
- **Timeout**: 10 minutes (configurable via `DIARIZATION_TIMEOUT`)
- **Retry Policy**: Extended model loading policy
- **Retries**: 3 attempts with exponential backoff

### Speaker Assignment Activity
- **Timeout**: 5 minutes (configurable via `SPEAKER_ASSIGNMENT_TIMEOUT`)
- **Retry Policy**: Minimal retry policy
- **Retries**: 2 attempts with quick backoff

## Configuration

All retry policies can be configured via environment variables:

```bash
# Basic retry configuration
TEMPORAL_MAX_ATTEMPTS=3
TEMPORAL_INITIAL_INTERVAL=5
TEMPORAL_BACKOFF_COEFFICIENT=2.0
TEMPORAL_MAX_INTERVAL=300

# Activity timeouts (in minutes)
TRANSCRIPTION_TIMEOUT=30
ALIGNMENT_TIMEOUT=10
DIARIZATION_TIMEOUT=10
SPEAKER_ASSIGNMENT_TIMEOUT=5
```

## Monitoring and Logging

- All retry attempts are logged with context
- Activity execution times are tracked
- Workflow progress is monitored at each step
- Final failures are logged with attempt counts

## Best Practices

1. **Model Loading**: Allow longer initial intervals for model downloads
2. **GPU Memory**: Use shorter, more frequent retries for memory issues
3. **Network Issues**: Implement exponential backoff for network problems
4. **Authentication**: Mark as non-retryable to avoid unnecessary attempts
5. **Monitoring**: Track retry patterns to optimize policies

## Error Handling Flow

```
Activity Error Occurs
         ↓
Error Classification
         ↓
Create ApplicationError
         ↓
Temporal Retry Logic
         ↓
Success or Final Failure
```

## Troubleshooting

### High Retry Rates
- Check model availability and network connectivity
- Verify HF_TOKEN is valid and has required permissions
- Monitor GPU memory usage patterns

### Timeout Issues
- Increase activity timeouts for large files
- Consider splitting large processing tasks
- Monitor resource utilization during processing

### Authentication Failures
- Verify HF_TOKEN environment variable
- Check model permissions on Hugging Face Hub
- Ensure terms of service are accepted for required models
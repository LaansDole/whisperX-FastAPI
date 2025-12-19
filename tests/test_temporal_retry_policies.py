"""Tests for temporal retry policies and error handling."""

import pytest
from datetime import timedelta
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

from app.temporal.config import (
    get_default_retry_policy,
    get_model_loading_retry_policy,
    get_gpu_memory_retry_policy,
)
from app.temporal.error_handler import TemporalErrorHandler


class TestTemporalRetryPolicies:
    """Test cases for temporal retry policy configuration."""

    def test_default_retry_policy(self):
        """Test default retry policy configuration."""
        policy = get_default_retry_policy()
        assert isinstance(policy, RetryPolicy)
        assert policy.initial_interval == timedelta(seconds=5)
        assert policy.backoff_coefficient == 2.0
        assert policy.maximum_attempts == 3

    def test_model_loading_retry_policy(self):
        """Test model loading retry policy configuration."""
        policy = get_model_loading_retry_policy()
        assert isinstance(policy, RetryPolicy)
        assert policy.initial_interval == timedelta(seconds=10)
        assert policy.backoff_coefficient == 1.5
        assert policy.maximum_attempts == 5

    def test_gpu_memory_retry_policy(self):
        """Test GPU memory retry policy configuration."""
        policy = get_gpu_memory_retry_policy()
        assert isinstance(policy, RetryPolicy)
        assert policy.initial_interval == timedelta(seconds=15)
        assert policy.backoff_coefficient == 1.2
        assert policy.maximum_attempts == 3


class TestTemporalErrorHandler:
    """Test cases for temporal error handling."""

    def test_classify_authentication_error(self):
        """Test classification of authentication errors as non-retryable."""
        error = RuntimeError("401 authentication failed")
        is_retryable, category = TemporalErrorHandler.classify_error(error)
        assert not is_retryable
        assert category == "authentication"

    def test_classify_network_error(self):
        """Test classification of network errors as retryable."""
        error = RuntimeError("network connection failed")
        is_retryable, category = TemporalErrorHandler.classify_error(error)
        assert is_retryable
        assert category == "network"

    def test_classify_gpu_memory_error(self):
        """Test classification of GPU memory errors as retryable."""
        error = RuntimeError("CUDA out of memory")
        is_retryable, category = TemporalErrorHandler.classify_error(error)
        assert is_retryable
        assert category == "gpu_memory"

    def test_classify_configuration_error(self):
        """Test classification of configuration errors as non-retryable."""
        error = ValueError("Missing required configuration")
        is_retryable, category = TemporalErrorHandler.classify_error(error)
        assert not is_retryable
        assert category == "configuration"

    def test_create_application_error_retryable(self):
        """Test creation of retryable ApplicationError."""
        error = RuntimeError("network timeout")
        app_error = TemporalErrorHandler.create_application_error(error, "Test context")
        assert isinstance(app_error, ApplicationError)
        assert not app_error.non_retryable
        assert "Test context" in str(app_error)

    def test_create_application_error_non_retryable(self):
        """Test creation of non-retryable ApplicationError."""
        error = RuntimeError("authentication failed")
        app_error = TemporalErrorHandler.create_application_error(error, "Test context")
        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable

    def test_force_non_retryable(self):
        """Test forcing an error to be non-retryable."""
        error = RuntimeError("network timeout")  # Normally retryable
        app_error = TemporalErrorHandler.create_application_error(error, "Test context", force_non_retryable=True)
        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable


if __name__ == "__main__":
    pytest.main([__file__])

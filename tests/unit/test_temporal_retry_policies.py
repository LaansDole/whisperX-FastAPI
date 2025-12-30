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

    def test_create_application_error_string_default_retryable(self):
        """Test that string errors default to retryable (True) matching Temporal SDK."""
        app_error = TemporalErrorHandler.create_application_error("Network timeout", "Test context")
        assert isinstance(app_error, ApplicationError)
        assert not app_error.non_retryable  # Should be retryable by default
        assert "Network timeout" in str(app_error)

    def test_create_application_error_string_explicit_non_retryable(self):
        """Test that string errors can be explicitly marked non-retryable."""
        app_error = TemporalErrorHandler.create_application_error(
            "Invalid configuration", "Test context", retryable=False
        )
        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable  # Explicitly non-retryable
        assert "Invalid configuration" in str(app_error)

    def test_create_application_error_explicit_retryable_override(self):
        """Test that explicit retryable parameter overrides auto-classification."""
        # ValueError is normally non-retryable, but explicit override makes it retryable
        error = ValueError("Validation error")
        app_error = TemporalErrorHandler.create_application_error(error, "Test context", retryable=True)
        assert isinstance(app_error, ApplicationError)
        assert not app_error.non_retryable  # Explicitly made retryable
        assert "Validation error" in str(app_error)  # Check message content

    def test_temporal_non_retryable_error_behavior(self):
        """Test that non-retryable errors in activities won't be retried by Temporal.

        This test simulates an activity raising a ValueError (non-retryable error)
        and verifies that:
        1. The error is properly classified as non-retryable
        2. ApplicationError is created with non_retryable=True
        3. Temporal SDK would NOT retry this error
        """

        # Simulate a Temporal activity that raises a ValueError
        def mock_activity_with_validation_error():
            """Simulated activity that validates input and fails."""
            patient_id = None  # Invalid input
            if not patient_id:
                error = ValueError("patient_id is required")
                # Activity should raise ApplicationError via error handler
                raise TemporalErrorHandler.create_application_error(
                    error,
                    "Activity: validate_patient_input",
                    retryable=None,  # Let auto-classification decide
                )

        # Execute the simulated activity and catch the error
        try:
            mock_activity_with_validation_error()
            assert False, "Expected ApplicationError to be raised"
        except ApplicationError as app_error:
            # Verify the error properties
            assert app_error.non_retryable is True, "ValueError should be non-retryable"
            assert "patient_id is required" in str(app_error)
            assert "Activity: validate_patient_input" in str(app_error)

            # Verify this matches Temporal's retry behavior expectations
            # When non_retryable=True, Temporal will:
            # - NOT retry the activity
            # - Fail the workflow immediately
            # - Mark the activity as FAILED (not RETRYING)
            print("\n✓ Non-retryable error correctly configured:")
            print(f"  - non_retryable: {app_error.non_retryable}")
            print("  - Temporal behavior: Will NOT retry")
            print("  - Activity status: FAILED (permanent)")

    def test_temporal_retryable_error_behavior(self):
        """Test that retryable errors in activities WILL be retried by Temporal.

        This test simulates an activity raising a network error (retryable)
        and verifies that:
        1. The error is properly classified as retryable
        2. ApplicationError is created with non_retryable=False
        3. Temporal SDK WOULD retry this error
        """

        # Simulate a Temporal activity that encounters a transient network error
        def mock_activity_with_network_error():
            """Simulated activity that fails due to network timeout."""
            error = RuntimeError("Connection timeout to LM Studio")
            # Activity should raise ApplicationError via error handler
            raise TemporalErrorHandler.create_application_error(
                error,
                "Activity: call_lm_studio",
                retryable=None,  # Let auto-classification decide
            )

        # Execute the simulated activity and catch the error
        try:
            mock_activity_with_network_error()
            assert False, "Expected ApplicationError to be raised"
        except ApplicationError as app_error:
            # Verify the error properties
            assert app_error.non_retryable is False, "Network errors should be retryable"
            assert "Connection timeout" in str(app_error)
            assert "Activity: call_lm_studio" in str(app_error)

            # Verify this matches Temporal's retry behavior expectations
            # When non_retryable=False, Temporal will:
            # - Retry the activity based on retry policy
            # - Apply exponential backoff
            # - Mark the activity as RETRYING (not FAILED)
            print("\n✓ Retryable error correctly configured:")
            print(f"  - non_retryable: {app_error.non_retryable}")
            print("  - Temporal behavior: WILL retry with backoff")
            print("  - Activity status: RETRYING (transient failure)")


if __name__ == "__main__":
    pytest.main([__file__])

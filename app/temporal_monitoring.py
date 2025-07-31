"""Temporal monitoring and metrics utilities."""

import logging
import time
from typing import Dict, Any
from contextlib import asynccontextmanager
from temporalio import workflow, activity

logger = logging.getLogger(__name__)


class TemporalMetrics:
    """Utility class for temporal metrics and monitoring."""
    
    @staticmethod
    @asynccontextmanager
    async def activity_timer(activity_name: str, audio_path: str = ""):
        """Context manager to time activity execution."""
        start_time = time.time()
        logger.info(f"Starting {activity_name} for {audio_path}")
        
        try:
            yield
            duration = time.time() - start_time
            logger.info(f"Completed {activity_name} for {audio_path} in {duration:.2f}s")
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed {activity_name} for {audio_path} after {duration:.2f}s: {e}")
            raise
    
    @staticmethod
    def log_workflow_progress(step: str, audio_path: str, additional_info: Dict[str, Any] = None):
        """Log workflow progress with structured information."""
        info = {"step": step, "audio_path": audio_path}
        if additional_info:
            info.update(additional_info)
        
        logger.info(f"Workflow progress: {info}")
    
    @staticmethod
    def log_retry_attempt(activity_name: str, attempt: int, error: str):
        """Log retry attempts for monitoring."""
        logger.warning(f"Retry attempt {attempt} for {activity_name}: {error}")
    
    @staticmethod 
    def log_final_failure(activity_name: str, total_attempts: int, final_error: str):
        """Log final failure after all retries exhausted."""
        logger.error(f"Activity {activity_name} failed after {total_attempts} attempts: {final_error}")


def monitor_activity(activity_name: str = None):
    """
    Decorator to add monitoring to activity functions.
    
    Args:
        activity_name: Optional custom name for the activity
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = activity_name or func.__name__
            audio_path = args[0] if args and isinstance(args[0], str) else "unknown"
            
            with TemporalMetrics.activity_timer(name, audio_path):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_workflow(workflow_name: str = None):
    """
    Decorator to add monitoring to workflow functions.
    
    Args:
        workflow_name: Optional custom name for the workflow  
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            name = workflow_name or func.__name__
            start_time = time.time()
            
            # Extract audio_path from args if available
            audio_path = "unknown"
            if len(args) > 1 and isinstance(args[1], str):
                audio_path = args[1]
            
            logger.info(f"Starting workflow {name} for {audio_path}")
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Completed workflow {name} for {audio_path} in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed workflow {name} for {audio_path} after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator
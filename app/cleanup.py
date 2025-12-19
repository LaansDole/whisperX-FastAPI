"""
Resource cleanup utilities for server shutdown.
"""

import atexit
import threading
import gc
import logging
from .warnings_filter import filter_warnings

logger = logging.getLogger(__name__)


def cleanup_threads():
    """Clean up threads on server shutdown."""
    try:
        logger.info("Cleaning up threads on shutdown...")
        thread_count = 0
        for thread in threading.enumerate():
            # Clean up threads related to audio processing
            if any(
                keyword in thread.name.lower() for keyword in ["whisper", "audio", "speech", "transcrib", "process"]
            ):
                try:
                    logger.debug(f"Joining thread: {thread.name}")
                    thread.join(timeout=1.0)
                    thread_count += 1
                except Exception as e:
                    logger.debug(f"Could not join thread {thread.name}: {e}")

        logger.info(f"Cleaned up {thread_count} threads")
    except Exception as e:
        logger.error(f"Error during thread cleanup: {e}")


def cleanup_resources():
    """Clean up resources and force garbage collection."""
    try:
        logger.info("Cleaning up resources on shutdown...")

        # Force garbage collection
        objects_collected = gc.collect()
        logger.debug(f"Garbage collection: {objects_collected} objects collected")

        # Clean up torch resources if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error clearing CUDA cache: {e}")

        logger.info("Resource cleanup completed")
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")


def setup_cleanup_handlers():
    """Set up cleanup handlers to run on server shutdown."""
    try:
        # Register cleanup handlers
        atexit.register(cleanup_resources)
        atexit.register(cleanup_threads)

        # Apply warning filters
        filter_warnings()

        logger.debug("Cleanup handlers registered successfully")
    except Exception as e:
        logger.warning(f"Could not register cleanup handlers: {e}")
        print(f"⚠️  Warning: Could not register cleanup handlers: {e}")
        print("   Resource cleanup will not be available")


if __name__ == "__main__":
    # Test the cleanup functions
    setup_cleanup_handlers()
    print("✅ Cleanup handlers registered successfully")
    print("   Thread cleanup: Will join audio/processing threads on shutdown")
    print("   Resource cleanup: Will force GC and clean CUDA cache on shutdown")
    print("   Warning filters: Will suppress resource tracker warnings")

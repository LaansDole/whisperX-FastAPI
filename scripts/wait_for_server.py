"""
Wait for FastAPI server to be ready before proceeding.
This script polls the FastAPI server health endpoint until it responds successfully.
"""

import sys
import time
import urllib.request
import urllib.error


def wait_for_server(
    url: str = "http://localhost:8000/health",
    max_attempts: int = 30,
    delay: float = 1.0,
) -> bool:
    """Wait for FastAPI server to be ready by polling the health endpoint.

    Args:
        url: The health endpoint URL to poll.
        max_attempts: Maximum number of attempts before timing out.
        delay: Delay in seconds between attempts.

    Returns:
        True if server is ready, False if timeout occurred.
    """
    print(f"Waiting for FastAPI server at {url}...")

    for attempt in range(max_attempts):
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    print(f"[OK] FastAPI server is ready (attempt {attempt + 1})")
                    return True
        except urllib.error.URLError:
            pass
        except urllib.error.HTTPError as e:
            # Server is responding but returned an error - still means it's up
            if e.code < 500:
                print(f"[OK] FastAPI server is responding (attempt {attempt + 1})")
                return True
        except Exception:
            pass

        print(f"Attempt {attempt + 1}/{max_attempts} - Server not ready yet...")
        time.sleep(delay)

    print("[FAIL] FastAPI server failed to start within timeout period")
    return False


if __name__ == "__main__":
    success = wait_for_server()
    sys.exit(0 if success else 1)

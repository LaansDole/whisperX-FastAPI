"""
Wait for Temporal worker to be ready before proceeding.
This script checks if the Temporal worker is responsive.
"""

import sys
import os
import time
import socket

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.temporal.config import TemporalConfig

def wait_for_worker(max_attempts=30, delay=2):
    """Wait for Temporal worker to be ready"""
    print("Waiting for Temporal worker to be ready...")
    
    host, port = TemporalConfig.TEMPORAL_SERVER_URL.split(':')
    port = int(port)
    
    for attempt in range(max_attempts):
        try:
            # Try to connect to Temporal server via socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"[OK] Temporal server is reachable (attempt {attempt + 1})")
                # Additional check to see if worker process is running
                import subprocess
                try:
                    result = subprocess.run(['pgrep', '-f', 'temporal.worker'],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print("[OK] Temporal worker process is running")
                        return True
                    else:
                        print("[OK] Temporal server reachable but worker not yet started")
                except Exception:
                    pass
        except Exception:
            pass
        
        print(f"Attempt {attempt + 1}/{max_attempts} - Temporal worker not ready yet...")
        time.sleep(delay)
    
    print("[FAIL] Temporal worker failed to start within timeout period")
    return False

if __name__ == "__main__":
    success = wait_for_worker()
    sys.exit(0 if success else 1)
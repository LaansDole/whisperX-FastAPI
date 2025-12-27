import subprocess
import os
import signal
import sys

# Set flag to clear database on server startup only
os.environ["DB_FRESH_START"] = "true"

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import cleanup handlers
try:
    from app.cleanup import setup_cleanup_handlers
    setup_cleanup_handlers()
    print("Resource cleanup handlers registered")
except ImportError as e:
    print(f"Warning: Could not import cleanup handlers: {e}")
    print("   Resource cleanup will not be available")

try:
    from app.warnings_filter import filter_warnings
    filter_warnings()
except ImportError as e:
    print(f"Warning: Could not import warning filters: {e}")
    print("   Warning filters will not be available")

# Function to kill any process using a specific port
def kill_port(port):
    """Kill any process using the specified port"""
    try:
        print(f"Checking for processes using port {port}...")
        # Find processes using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"Killing process {pid} using port {port}")
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except (ProcessLookupError, ValueError):
                        print(f"Process {pid} not found or invalid PID")
            print(f"All processes on port {port} have been terminated")
        else:
            print(f"No processes found using port {port}")
            
    except Exception as e:
        print(f"Error checking port {port}: {e}")

def main():
    """Start the FastAPI server"""
    try:
        # Kill any existing processes on port 8000
        kill_port(8000)
        
        # Check if we're already in the whisperX-FastAPI directory
        current_dir = os.path.basename(os.getcwd())
        if current_dir != "whisperX-FastAPI":
            os.chdir("whisperX-FastAPI")
            print("Changed directory to whisperX-FastAPI")
        else:
            print("Already in whisperX-FastAPI directory")
            
        # Define the command to run (using port 8000)
        command = ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", 
                   "--log-config", "app/uvicorn_log_conf.yaml", "--log-level", "info"]

        print("Starting FastAPI server...")
        print(f"Running command: {' '.join(command)}")
        
        # Run the server directly in the foreground
        subprocess.run(command)

    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting FastAPI server: {e}")

if __name__ == "__main__":
    main()
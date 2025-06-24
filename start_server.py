import subprocess
import os
import time
import requests
import signal
import threading
import sys

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
                        time.sleep(1)  # Give it a moment to terminate gracefully
                        # If still running, force kill
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # Process already terminated
                    except (ProcessLookupError, ValueError):
                        print(f"Process {pid} not found or invalid PID")
            print(f"All processes on port {port} have been terminated")
        else:
            print(f"No processes found using port {port}")
            
    except Exception as e:
        print(f"Error checking port {port}: {e}")

# Function to stream output from a process
def stream_output(process, prefix=""):
    """Stream output from a process to stdout in real-time"""
    for line in iter(process.stdout.readline, b''):
        if line:
            sys.stdout.write(f"{prefix} {line.decode('utf-8')}")
            sys.stdout.flush()
    
    if process.stderr:
        for line in iter(process.stderr.readline, b''):
            if line:
                sys.stderr.write(f"{prefix} ERROR: {line.decode('utf-8')}")
                sys.stderr.flush()

# Function to start FastAPI server in a monitored process
def start_fastapi_background():
    print("Attempting to start FastAPI server in monitored process...")
    try:
        # Kill any existing processes on port 8000
        kill_port(8000)
        
        # Check if we're already in the whisperX-FastAPI directory
        current_dir = os.path.basename(os.getcwd())
        if current_dir != "whisperX-FastAPI":
            os.chdir("whisperX-FastAPI")
            print(f"Changed directory to whisperX-FastAPI")
        else:
            print("Already in whisperX-FastAPI directory")
            
        # Define the command to run (using port 8000)
        command = ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", 
                   "--log-config", "app/uvicorn_log_conf.yaml", "--log-level", "info"]

        # Start the process with output capturing
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1  # Line buffered
        )

        # Start threads to stream the output without blocking
        stdout_thread = threading.Thread(target=stream_output, args=(process, "SERVER:"))
        stdout_thread.daemon = True
        stdout_thread.start()

        print(f"FastAPI process started with PID: {process.pid}")
        return process

    except Exception as e:
        print(f"Error starting FastAPI process: {e}")
        return None

# Start the background process
fastapi_process = start_fastapi_background()

# Give the server a moment to start
print("Waiting 20 seconds for the server to start...")
time.sleep(20)

global server_url
# Now check if the server is reachable
server_url = "http://0.0.0.0:8000"
print(f"Attempting to connect to FastAPI server at {server_url}...")

try:
    response = requests.get(server_url, timeout=20)
    print(f"Successfully connected to {server_url}. Status code: {response.status_code}")
    if response.status_code in [200, 404, 422]:
        print("FastAPI server appears to be running.")
    else:
        print("Received unexpected status code. Server might be running but not as expected.")

except requests.exceptions.ConnectionError:
    print(f"Error: Could not connect to {server_url}. The server might not be running or is not accessible.")
except requests.exceptions.Timeout:
    print(f"Error: Request to {server_url} timed out.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Store the process globally if needed for later shutdown
globals()['fastapi_background_process'] = fastapi_process

# Keep the script running to continue streaming server output
print("\nServer is running. Press Ctrl+C to stop.")
try:
    # Wait for the server process to complete or user to interrupt
    while fastapi_process.poll() is None:
        time.sleep(1)
    
    # If we get here, the server has stopped
    exit_code = fastapi_process.returncode
    print(f"\nServer process has exited with code {exit_code}")
    
except KeyboardInterrupt:
    print("\nShutting down server...")
    try:
        os.killpg(os.getpgid(fastapi_process.pid), signal.SIGTERM)
        print("Server stopped.")
    except Exception as e:
        print(f"Error stopping server: {e}")
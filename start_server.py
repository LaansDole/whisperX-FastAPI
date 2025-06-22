import subprocess
import os
import time
import requests
import signal

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
            
    except FileNotFoundError:
        print("lsof command not found, trying alternative method...")
        # Alternative method using netstat and kill
        try:
            result = subprocess.run(
                ["netstat", "-tlnp"], 
                capture_output=True, 
                text=True
            )
            lines = result.stdout.split('\n')
            for line in lines:
                if f":{port} " in line and "LISTEN" in line:
                    parts = line.split()
                    if len(parts) > 6:
                        pid_program = parts[6]
                        if '/' in pid_program:
                            pid = pid_program.split('/')[0]
                            try:
                                print(f"Killing process {pid} using port {port}")
                                os.kill(int(pid), signal.SIGTERM)
                                time.sleep(1)
                                try:
                                    os.kill(int(pid), signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                            except (ProcessLookupError, ValueError):
                                print(f"Process {pid} not found or invalid PID")
        except Exception as e:
            print(f"Could not kill processes on port {port}: {e}")
    except Exception as e:
        print(f"Error checking port {port}: {e}")

# Function to start FastAPI server in a background process
def start_fastapi_background():
    print("Attempting to start FastAPI server in background using subprocess...")
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
            
        # Define the command to run (using port 8000 instead of 8001)
        # Use 'exec' to replace the current shell process with uvicorn,
        # and run in the background with '&'
        # Ensure the virtual environment is sourced
        command = "uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-config app/uvicorn_log_conf.yaml --log-level info &"

        # Start the subprocess
        # We don't need to capture stdout/stderr if we want it to run truly in background
        # If we need to debug, we might temporarily remove '&' and check output
        process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)

        print(f"FastAPI background process started with PID: {process.pid}")
        return process

    except Exception as e:
        print(f"Error starting FastAPI background process: {e}")
        return None

# Start the background process
fastapi_process = start_fastapi_background()

# Give the server a moment to start
time.sleep(20)

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
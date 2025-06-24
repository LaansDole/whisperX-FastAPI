import os
import signal
import subprocess
import threading
import time
from google.colab.output import serve_kernel_port_as_iframe

# Check if we're already in the whisperX-FastAPI directory
current_dir = os.path.basename(os.getcwd())
if current_dir != "whisperX-FastAPI":
    os.chdir("whisperX-FastAPI")
    print(f"Changed directory to whisperX-FastAPI")
else:
    print("Already in whisperX-FastAPI directory")

# --- Configuration ---
PORT = 8000
LOG_CONFIG_PATH = "app/uvicorn_log_conf.yaml"
APP_MODULE = "app.main:app"

# --- Global variable to hold the server process ---
server_process = None

def kill_port(port):
    """Kills any process listening on the given port."""
    print(f"Checking for and terminating any process on port {port}...")
    try:
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        if result.stdout:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"Killed process {pid} on port {port}.")
                except (ProcessLookupError, ValueError):
                    pass  # Process already gone
    except FileNotFoundError:
        print("`lsof` command not found. Skipping port clearing.")
    except Exception as e:
        print(f"An error occurred while trying to kill port {port}: {e}")

def start_server():
    """Starts the Uvicorn server in a background thread."""
    global server_process

    # Ensure we are in the correct directory
    if os.path.basename(os.getcwd()) != "whisperX-FastAPI":
        os.chdir("whisperX-FastAPI")
        print("Changed directory to whisperX-FastAPI")

    # First, ensure the port is free
    kill_port(PORT)

    # Command to start Uvicorn
    command = [
        "uvicorn",
        APP_MODULE,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--log-config", LOG_CONFIG_PATH,
        "--log-level", "info"
    ]

    # Start the server as a background process
    print("Starting FastAPI server...")
    server_process = subprocess.Popen(command)
    print(f"Server process started with PID: {server_process.pid}")

    # Wait a moment for the server to initialize
    time.sleep(5)

    # Expose the port to a public URL
    print(f"Exposing port {PORT} as an iframe...")
    serve_kernel_port_as_iframe(port=PORT, height=800)

def stop_server():
    """Stops the background Uvicorn server."""
    global server_process
    if server_process:
        print(f"Stopping server process with PID: {server_process.pid}...")
        server_process.terminate()
        try:
            # Wait for the process to terminate
            server_process.wait(timeout=10)
            print("Server stopped successfully.")
        except subprocess.TimeoutExpired:
            print("Server did not terminate gracefully. Forcing shutdown...")
            server_process.kill()
            print("Server forced to shut down.")
        server_process = None
    else:
        print("Server is not running.")

# --- Main execution ---
if __name__ == "__main__":
    try:
        start_server()
        # The server is running in the background.
        # The script will keep running, allowing the server to stay active.
        # To stop the server, you would call stop_server() in another cell.
        print("\nServer is running in the background.")
        print("To stop the server, call the stop_server() function.")
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down server...")
        stop_server()
        print("Shutdown complete.")

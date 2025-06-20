#!/usr/bin/env python3
import json

# Load the notebook
with open('notebooks/whisperx_fastapi_colab.ipynb', 'r') as f:
    nb = json.load(f)

# Fixed FastAPI code
fixed_code = """# Function to run FastAPI in the background
import subprocess
import time
import threading
import os
import requests
import IPython.display as display
from IPython.display import clear_output

# Start FastAPI server in a separate process
def start_fastapi_server():
    print("Starting FastAPI server...")
    try:
        # Change to the project directory
        os.chdir("whisperX-FastAPI")
        
        # Command to start the FastAPI application with virtual environment
        command = "bash -c 'source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-config app/uvicorn_log_conf.yaml'"
        
        # Start the process
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            preexec_fn=os.setsid  # Create new process group
        )
        
        print("FastAPI server process started. Process ID:", process.pid)
        time.sleep(3)  # Give some time for the server to start
        
        # Change back to original directory
        os.chdir("..")
        
        return process
        
    except Exception as e:
        print(f"Error in start_fastapi_server: {e}")
        # Make sure we change back to original directory
        try:
            os.chdir("..")
        except:
            pass
        return None

# Wait for FastAPI HTTP API to be ready
def wait_for_fastapi(timeout=60):
    print("Waiting for FastAPI to become ready...")
    for i in range(timeout):
        try:
            # Try both root endpoint and docs endpoint
            response = requests.get("http://localhost:8000/", timeout=2)
            if response.status_code in [200, 404, 422]:  # 422 is expected for root endpoint
                print(f"✅ FastAPI is ready! (after {i+1}s)")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"⏳ Waiting for FastAPI to start... {i+1}s")
        time.sleep(1)
    
    print("❌ FastAPI did not start within timeout period")
    return False

# Start the services
fastapi_process = None

try:
    print("=== Starting FastAPI Service ===")
    fastapi_process = start_fastapi_server()
    
    if fastapi_process:
        # Wait for the service to be ready
        if wait_for_fastapi():
            print("\\n✅ FastAPI service is running successfully!")
            print("📝 API Documentation: http://localhost:8000/docs")
            print("🔗 API Root: http://localhost:8000/")
            
            # Display clickable links in notebook
            display.display(display.HTML('''
                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; background-color: #f9fff9;">
                    <h3>🎉 FastAPI is Ready!</h3>
                    <p><strong>API Documentation:</strong> <a href="http://localhost:8000/docs" target="_blank">http://localhost:8000/docs</a></p>
                    <p><strong>API Root:</strong> <a href="http://localhost:8000/" target="_blank">http://localhost:8000/</a></p>
                    <p><em>Note: These links work if you're running locally. For Colab, you'll need to use the Cloudflare tunnel.</em></p>
                </div>
            '''))
        else:
            print("❌ FastAPI failed to start properly")
            if fastapi_process:
                fastapi_process.terminate()
    else:
        print("❌ Failed to start FastAPI process")
        
except Exception as e:
    print(f"❌ Error starting FastAPI service: {str(e)}")
    if fastapi_process:
        try:
            fastapi_process.terminate()
        except:
            pass

# Store the process globally so it can be accessed later
globals()["fastapi_process"] = fastapi_process"""

# Find and replace the FastAPI server cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'start_fastapi_server' in source and 'def start_fastapi_server' in source:
            print(f'Fixing FastAPI server code in Cell {i+1}')
            
            # Update the cell with fixed code
            cell['source'] = fixed_code.split('\n')
            print(f'Updated Cell {i+1} with corrected FastAPI startup code')
            break

# Save the updated notebook
with open('notebooks/whisperx_fastapi_colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Notebook updated successfully with fixed FastAPI startup!')

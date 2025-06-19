# WhisperX FastAPI Notebooks

This directory contains notebooks for running and using the WhisperX FastAPI project in different environments.

## whisperx_fastapi_colab.ipynb

This notebook allows you to run the WhisperX FastAPI project on Google Colab to leverage their GPU resources. It provides a complete environment for running the API service and exposing it through a Cloudflare tunnel for external access.

### Features

- Automatically sets up the complete WhisperX FastAPI environment on Google Colab
- Uses Colab's GPU for fast speech-to-text processing
- Exposes the API through a Cloudflare tunnel for external access
- Includes examples for uploading and processing audio files
- Provides task status checking and monitoring capabilities

### Requirements

- A Google account with access to Google Colab
- A Hugging Face account with an API token
- Internet connection for the Cloudflare tunnel

### How to Use

1. Open the notebook in Google Colab:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click on "File" > "Open notebook"
   - Select the "GitHub" tab
   - Enter the repository URL: `https://github.com/pavelzbornik/whisperX-FastAPI`
   - Select the notebook: `notebooks/whisperx_fastapi_colab.ipynb`

2. Make sure you're using a GPU runtime:
   - Click on "Runtime" > "Change runtime type"
   - Select "GPU" as the hardware accelerator
   - Click "Save"

3. Run the notebook cells in order:
   - The notebook will install all necessary dependencies
   - It will clone the repository and set up the environment
   - You'll be prompted to enter your Hugging Face token
   - The FastAPI service will be started and exposed through a Cloudflare tunnel

4. Access the API:
   - The notebook will display a clickable link to the API documentation
   - You can use the API from any device with internet access
   - Example code for uploading and processing audio is included in the notebook

5. When done, use the shutdown cells to properly terminate the services and clean up resources.

### Troubleshooting

- If the Cloudflare tunnel fails to establish, try running the cell again
- If the API is unresponsive, check the GPU usage and consider restarting the runtime
- For errors related to Hugging Face token, verify that your token has the necessary permissions

### Notes

- The notebook uses the `tiny` Whisper model by default, but you can specify a larger model when prompted
- Larger models will provide better transcription quality but will use more GPU memory and may be slower
- The SQLite database is stored in memory and will be lost when the Colab session ends
- For processing very large audio files, consider using smaller chunks or a larger Colab instance

For more information about WhisperX and its capabilities, see the [main README](../README.md) and the [WhisperX documentation](https://github.com/m-bain/whisperX).

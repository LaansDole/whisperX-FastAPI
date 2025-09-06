# whisperX REST API

The whisperX API is a production-ready tool for enhancing and analyzing audio content using advanced speech processing technologies. This API provides a comprehensive suite of services for processing audio and video files, including transcription, alignment, diarization, and combining transcript with diarization results.

## System Architecture

The whisperX-FastAPI system is built with a modern, scalable architecture that leverages Temporal for workflow orchestration and provides robust error handling and monitoring capabilities.

```mermaid
graph TB
    %% Client Layer
    Client[Client Applications]
    WebUI[Web UI - Swagger/OpenAPI]
    
    %% API Gateway Layer
    subgraph "FastAPI Application"
        Router[API Routers]
        STT[Speech-to-Text Routes]
        Services[Individual Services]
        Tasks[Task Management]
        RAG[RAG Chatbot - Optional]
        Health[Health Endpoints]
        Middleware[Trace Middleware]
    end
    
    %% Core Processing Layer
    subgraph "WhisperX Processing Engine"
        ModelManager[Model Manager]
        WhisperXService[WhisperX Services]
        AudioProcessor[Audio Processor]
        Transcriber[Transcriber]
        Aligner[Aligner]
        Diarizer[Diarizer]
        Combiner[Result Combiner]
    end
    
    %% Workflow Orchestration Layer
    subgraph "Temporal Workflow System"
        TemporalClient[Temporal Client]
        WorkflowEngine[Workflow Engine]
        Activities[Activity Workers]
        RetryPolicies[Retry Policies]
        Monitoring[Workflow Monitoring]
    end
    
    %% Storage and Model Layer
    subgraph "Storage & Models"
        ModelCache[Model Cache<br/>~/.cache/huggingface]
        TorchCache[PyTorch Cache<br/>~/.cache/torch]
        FileStorage[Temporary File Storage]
        HuggingFace[Hugging Face Hub]
    end
    
    %% Infrastructure Layer
    subgraph "Infrastructure"
        Docker[Docker Containers]
        GPU[GPU Support<br/>CUDA 12.8+]
        CPU[CPU Fallback]
        Logging[Structured Logging]
        HealthChecks[Health Monitoring]
    end
    
    %% Connections
    Client --> Router
    WebUI --> Router
    Router --> STT
    Router --> Services
    Router --> Tasks
    Router --> RAG
    Router --> Health
    Router --> Middleware
    
    STT --> TemporalClient
    Services --> WhisperXService
    Tasks --> TemporalClient
    
    TemporalClient --> WorkflowEngine
    WorkflowEngine --> Activities
    Activities --> RetryPolicies
    Activities --> WhisperXService
    
    WhisperXService --> ModelManager
    WhisperXService --> AudioProcessor
    AudioProcessor --> Transcriber
    AudioProcessor --> Aligner
    AudioProcessor --> Diarizer
    AudioProcessor --> Combiner
    
    ModelManager --> ModelCache
    ModelManager --> TorchCache
    ModelManager --> HuggingFace
    
    WhisperXService --> FileStorage
    Activities --> Monitoring
    
    %% Infrastructure connections
    Docker -.-> GPU
    Docker -.-> CPU
    Router --> Logging
    Health --> HealthChecks
    
    %% Styling
    classDef clientLayer fill:#e1f5fe
    classDef apiLayer fill:#f3e5f5
    classDef processingLayer fill:#e8f5e8
    classDef workflowLayer fill:#fff3e0
    classDef storageLayer fill:#fce4ec
    classDef infraLayer fill:#f1f8e9
    
    class Client,WebUI clientLayer
    class Router,STT,Services,Tasks,RAG,Health,Middleware apiLayer
    class ModelManager,WhisperXService,AudioProcessor,Transcriber,Aligner,Diarizer,Combiner processingLayer
    class TemporalClient,WorkflowEngine,Activities,RetryPolicies,Monitoring workflowLayer
    class ModelCache,TorchCache,FileStorage,HuggingFace storageLayer
    class Docker,GPU,CPU,Logging,HealthChecks infraLayer
```

## Usage

### Local Run

1. **Create a virtual environment** and activate it.
2. **Install uv**: Follow the official instructions at [astral.sh/uv](https://astral.sh/uv).
3. **Install PyTorch**: Follow the official instructions at [pytorch.org](https://pytorch.org/).
4. **Install dependencies**:
   ```sh
   # For production dependencies
   make install-prod

   # For development dependencies
   make install-dev
   ```
5. **Create `.env` file**
    ```sh
    cp .env.example .env
    ```
6. **Run the application: FastAPI + Temporal**:

   To run temporal server locally without docker, you need to install the `temporal` CLI.
   Download and install the latest version for your system from the [official GitHub releases page](https://github.com/temporalio/cli/releases). You will need to move the `temporal` binary to a directory in your `PATH` (e.g., `/usr/local/bin`).

   Once installed, you can run a local temporal server.
   ```sh
   # Start the local temporal server
   make run-temporal-local

   # Stop the local temporal server
   make stop-temporal-local
   ```

   Now you can start the FastAPI server and the Temporal worker.
   ```sh
   # Start the FastAPI server
   make run-local

   # In a separate terminal, start the Temporal worker
   make run-worker-local
   ```

The API will be accessible at <http://127.0.0.1:8000>.

#### Model cache

The models used by whisperX are stored in `root/.cache`, if you want to avoid downloanding the models each time the container is starting you can store the cache in persistent storage. `docker-compose.gpu.yml` defines a volume `whisperx-models-cache` to store this cache.

- faster-whisper cache: `root/.cache/huggingface/hub`
- pyannotate and other models cache: `root/.cache/torch`

## Documentation

See the [WhisperX Documentation](https://github.com/m-bain/whisperX) for details on whisperX functions.

### Supported File Formats

#### Audio Files

- `.oga`, `.m4a`, `.aac`, `.wav`, `.amr`, `.wma`, `.awb`, `.mp3`, `.ogg`

#### Video Files

- `.wmv`, `.mkv`, `.avi`, `.mov`, `.mp4`

### Available Models

WhisperX supports a comprehensive range of model sizes and specialized variants:

#### Standard Models
- **Tiny**: `tiny`, `tiny.en` (~39MB, fastest)
- **Base**: `base`, `base.en` (~74MB, balanced)
- **Small**: `small`, `small.en` (~244MB, good accuracy)
- **Medium**: `medium`, `medium.en` (~769MB, better accuracy)
- **Large**: `large`, `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo` (~1550MB, best accuracy)

#### Distilled Models (Faster Inference)
- `distil-large-v2`, `distil-medium.en`, `distil-small.en`, `distil-large-v3`

#### Specialized Models
- **CrisperWhisper**: [`nyrahealth/faster_CrisperWhisper`](https://github.com/nyrahealth/CrisperWhisper) - Optimized for medical transcription

#### Model Configuration
- Set default model in `.env` using `WHISPER_MODEL=` (default: `small`)
- Models are automatically downloaded and cached on first use
- GPU models support `float16` and `float32` precision
- CPU models require `int8` quantization for optimal performance

## Troubleshooting

### Common Issues

1. **Model Download Failures**

   - Verify your internet connection.
   - Ensure the `HF_TOKEN` is correctly set in the `.env` file.
   - If you see an error like: `Error: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache`, try these solutions:
     
     a) Check your internet connection
     
     b) Verify your Hugging Face token has permission to access the models:
     ```sh
     # Test your token (replace YOUR_TOKEN with your actual token)
     curl -X GET https://huggingface.co/api/whoami -H "Authorization: Bearer YOUR_TOKEN"
     ```
     
     c) Try pre-downloading the model before running the service:
     ```sh
     # Example for downloading the base model
     python scripts/download_diarization_model.py
     ```

2. **Warnings Not Filtered**
   - Ensure the `FILTER_WARNING` environment variable is set to `true` in the `.env` file.

#### Workflow Monitoring

- **Temporal Web UI**: Access at `http://localhost:8233` for workflow visualization
- **Workflow Status API**: Track individual workflow progress via REST endpoints
- **Performance Metrics**: Built-in monitoring of processing times and success rates
- **Error Analytics**: Detailed error tracking and retry attempt logging

## Related

- [ahmetoner/whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)
- [alexgo84/whisperx-server](https://github.com/alexgo84/whisperx-server)
- [chinaboard/whisperX-service](https://github.com/chinaboard/whisperX-service)
- [tijszwinkels/whisperX-api](https://github.com/tijszwinkels/whisperX-api)

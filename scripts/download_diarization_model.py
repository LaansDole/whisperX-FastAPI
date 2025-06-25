import os
from huggingface_hub import snapshot_download

def download_model(model_name, cache_dir=None):
    """
    Downloads a model from the Hugging Face Hub.

    Args:
        model_name (str): The name of the model to download.
        cache_dir (str, optional): The directory to cache the model in. Defaults to None.
    """
    print(f"Downloading model: {model_name}")
    try:
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Model '{model_name}' downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model '{model_name}': {e}")

# Directly call the download function with the desired model name
download_model(model_name="pyannote/speaker-diarization-3.1", cache_dir="models/pyannote/speaker-diarization-3.1")
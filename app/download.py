import os
from google.cloud import storage

def download_model_from_bucket(bucket_name, model_files, save_dir):
    """
    Downloads specified files from a GCS bucket to a local directory.

    Args:
        bucket_name (str): The name of the GCS bucket.
        model_files (list): List of model file names to download.
        save_dir (str): Local directory to save the downloaded files.
    """
    # Initialize the Google Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket object
    bucket = storage_client.get_bucket(bucket_name)

    # Create the local directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    for file_name in model_files:
        # Construct the blob object for the file
        blob = bucket.blob(file_name)

        # Define the local file path
        local_file_path = os.path.join(save_dir, file_name)

        print(f"Downloading {file_name} to {local_file_path}...")

        # Download the file
        blob.download_to_filename(local_file_path)

    print("Model files downloaded successfully!")

if __name__ == "__main__":
    # Define the bucket name
    bucket_name = "summarizer-model-safetensors"  # Replace with your bucket name

    # Define the list of model files to download
    model_files = [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    ]

    # Define the directory to save the files - use current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "saved_model")  # Downloads to ./model directory

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Call the download function
    download_model_from_bucket(bucket_name, model_files, save_dir)
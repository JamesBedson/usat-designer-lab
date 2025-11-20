from google.cloud import storage
import os 
import shutil


def prepare_output_dir(yaml_path, bucket_name=None):
    yaml_base   = os.path.splitext(os.path.basename(yaml_path))[0]
    yaml_dir    = yaml_base
    os.makedirs(yaml_dir, exist_ok=True)

    yaml_dest = os.path.join(yaml_dir, f"{yaml_base}.yaml")

    # Copy YAML locally
    if not os.path.exists(yaml_dest):
        shutil.copy(yaml_path, yaml_dest)
        print(f"Copied YAML to destination: {yaml_dest}")

    return yaml_base, yaml_dir, yaml_dest


def download_yaml_to_directory(gcs_uri, dest_dir):
    
    if not is_gcs_path(gcs_uri):
        raise ValueError("Not a valid GCS URI: must start with 'gs://'")

    storage_client  = storage.Client()
    _, path         = gcs_uri.split("gs://", 1)
    
    bucket_name, *blob_parts = path.split("/")
    blob_name = "/".join(blob_parts)

    bucket  = storage_client.bucket(bucket_name)
    blob    = bucket.blob(blob_name)

    yaml_filename   = os.path.basename(blob_name)
    local_path      = os.path.join(dest_dir, yaml_filename)
    
    blob.download_to_filename(local_path)
    print(f"Downloaded YAML from {gcs_uri} to {local_path}")
    return local_path, bucket_name


def copy_local_yaml_to_directory(local_yaml_path, dest_dir):
    dest_path = os.path.join(dest_dir, os.path.basename(local_yaml_path))
    shutil.copy(local_yaml_path, dest_path)
    print(f"Copied YAML from {local_yaml_path} to {dest_path}")
    return dest_path


def upload_blob_to_gcs(local_file_path, bucket_name, destination_blob_name):

    client  = storage.Client()
    bucket  = client.bucket(bucket_name)
    blob    = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_file_path)
    print(f"Uploaded file {local_file_path} to gs://{bucket_name}/{destination_blob_name}")


def blob_exists(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()

def upload_directory_to_gcs(local_dir, 
                            bucket_name, 
                            gcs_prefix):
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for root, _, files in os.walk(local_dir):
        for file_name in files:
            local_path      = os.path.join(root, file_name)
            relative_path   = os.path.relpath(local_path, local_dir)
            gcs_path        = os.path.join(gcs_prefix, relative_path).replace("\\", "/")
            
            print(f"Uploading file {local_path} to gs://{bucket_name}/{gcs_path} ...")
            blob = bucket.blob(gcs_path)

            try:
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
            
            except Exception as e:
                print(f"Failed to upload {local_path}: {e}")


def is_gcs_path(path):
    return path.startswith("gs://")
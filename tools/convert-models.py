import torch
from google.cloud import storage
import os

version = "1"

models = {
  'wav2vec2_fairseq_base_ls960_asr_ls960': 'https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth',
}

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def download_file (url, filename):
  r = requests.get(url, allow_redirects=True, stream=True)
  with open(filename, 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024): 
        if chunk: f.write(chunk)

for name, url in models.items():
  fpath = os.path.basename(url)
  download_file(url, fpath)
  d = dict(torch.load(fpath))
  
  torch.save(d, fpath, _use_new_zipfile_serialization=True)
  upload_blob(
    "torchaudio-models",
    fpath,
     "/v" + version + "/models/" + fpath 
  )

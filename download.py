import zipfile
import wget
from os import path

if path.exists("./dataset/maestro-v3.0.0-midi"):
    print("Already downloaded")
else:
    url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip'
    filename = wget.download(url)
    with zipfile.ZipFile("maestro-v3.0.0-midi.zip", "r") as zip_ref:
        zip_ref.extractall("dataset/maestro-v3.0.0-midi")
        print("Dataset donwloaded")

import zipfile
import wget
from os import path
from utils.project_utils.logger_ import create_logger


logger = create_logger(__name__, log_to_file=False)

if path.exists("./dataset/maestro-v3.0.0-midi"):
    logger.info("Already downloaded")
else:
    url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip'
    filename = wget.download(url)
    with zipfile.ZipFile("maestro-v3.0.0-midi.zip", "r") as zip_ref:
        zip_ref.extractall("dataset/maestro-v3.0.0-midi")
        logger.info("Dataset donwloaded")

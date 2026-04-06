import os
import gdown
import zipfile

MODEL_DIR = "blip_flickr8k"

def download_model():

    if not os.path.exists(MODEL_DIR):

        file_id = "1GqAwogDYa_847QqgOVqMooakNKdSth-6"
        url = f"https://drive.google.com/uc?id={file_id}"

        output = "blip_finetuned.zip"

        gdown.download(url, output, quiet=False,fuzzy=True)

        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall()

        os.remove(output)

    return MODEL_DIR

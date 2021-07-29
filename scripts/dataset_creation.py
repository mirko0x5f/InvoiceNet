from pathlib import Path
from urllib.parse import urlparse
import json

from superai_dataclient.data_helper import DataHelper
import requests
import re, string
import io
from tqdm import tqdm
from PIL import Image

DATASET_PATH = "Aldi2.json"
DEST_DIR = "train/"
data_helper = DataHelper()
size = 128, 128
alphanumeric_pattern = re.compile('[\W_]+')


def fetch_data(data_url, image_path):
    signed_url = data_helper.get_data_url(data_url)
    response = requests.get(signed_url, allow_redirects=True)
    response.raise_for_status()

    image = Image.open(io.BytesIO(response.content))
    if image.size[0] < 128 or image.size[1] < 128:
        return False
    try:
        image.save(image_path)
    except OsError:
        print("Error during write, got a funky file!")
        return False
    return True


with open(DATASET_PATH, "r") as dataset_fp:
    dataset = json.load(dataset_fp)

for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
    try:
        labels = {}

        if not data["output"]["label"]:
            print(f"file #{idx} has no label")
            continue
            
        for label in data["output"]["label"]:
            # Strips non alphanumeric
            field_type = alphanumeric_pattern.sub('', label["field_type"])
            if field_type == "text":
                field_id = alphanumeric_pattern.sub('', label["field_id"])
                field_value = label["field_value"]
                if field_value and alphanumeric_pattern.sub('', field_value):
                    labels[field_id] = field_value

        if not labels:
            print(f"file #{idx} has no labels")
            continue
            
        image_url = data["input"]["image_url"]
        if not fetch_data(image_url, "train/" + str(idx) + ".jpg"):
            print(f"Error saving image, skipping {idx}")
            continue

        with open(DEST_DIR + str(idx) + ".json", "w") as json_fp:
            json.dump(labels, json_fp)
    except TypeError:
        print(f"file #{idx} raised typerror")

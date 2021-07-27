from pathlib import Path
from urllib.parse import urlparse
import json

from superai_dataclient.data_helper import DataHelper
import requests
from tqdm import tqdm

DATASET_PATH = "dataset.json"
data_helper = DataHelper()


def fetch_data(data_url, data_folder):
    signed_url = data_helper.get_data_url(data_url)
    response = requests.get(signed_url, allow_redirects=True)
    response.raise_for_status()

    url_path = Path(urlparse(data_url).path)
    data_path = Path(data_folder)  # , url_path.name)
    data_path.write_bytes(response.content)

    return data_path


with open(DATASET_PATH, "r") as dataset_fp:
    dataset = json.load(dataset_fp)

for idx, data in tqdm(enumerate(dataset)):
    image_url = data["input"]["image_url"]
    fetch_data(image_url, "train/" + str(idx) + ".jpg")
    label = data["output"]["label"][1]["field_value"]
    with open("train/" + str(idx) + ".json", "w") as json_fp:
        json.dump({"label": label}, json_fp)

import os
from typing import List, Tuple

import requests

from src import PROJECT_PATH


class Downloader:
    def __init__(self, gdrive_id: str = None, file_name: str = None, files: List[Tuple[str, str]] = None):
        """
        Constructor for downloading and loading the data file.
        :param str gdrive_id: Google Drive id
        :param str file_name: file name for saving
        :param files: a .json file, that contains the file name, and its Google Drive id
        """
        if gdrive_id is not None and file_name is not None:
            gdrive_link = "https://drive.google.com/uc?export=download&id="
            data_folder = os.path.join(PROJECT_PATH, "data")
            os.path.join(data_folder, "generated")
            os.makedirs(data_folder, exist_ok=True)
            file = os.path.join(data_folder, file_name)
            if not os.path.isfile(file):
                r = requests.get(gdrive_link + gdrive_id, allow_redirects=True)
                open(file, "wb").write(r.content)

        if files is not None:
            os.makedirs(os.path.join(PROJECT_PATH, "data", "generated"), exist_ok=True)
            gdrive_link = "https://drive.google.com/uc?export=download&id="
            for g_id in files:
                data_folder = os.path.join(PROJECT_PATH, "data")
                os.makedirs(data_folder, exist_ok=True)
                file = os.path.join(data_folder, g_id[1])
                if not os.path.isfile(file):
                    r = requests.get(gdrive_link + g_id[0], allow_redirects=True)
                    open(file, "wb").write(r.content)

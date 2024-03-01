import librosa
import os
import sys
import json
from src.logger import logging
from src.exception import CustomException

DATASET_PATH = os.path.join("data", "data")
JSON_PATH = os.path.join("data", "data.json")
SAMPLES_TO_CONSIDER = 22050


def prepare_dataset(dataset_path, json_path, n_mfcc = 13, hop_length = 512, n_fft = 2048):
    # data dictionanry

    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files" : [],
    }

    for i, (root, dirs, files) in enumerate(os.walk(dataset_path)):
        try:

            if root not in dataset_path:
                # update mappings
                category = root.split("\\")[-1]
                data["mappings"].append(category)
                print(f"Processing dataset with category {category}")
                # loop and extract MFCCs

                for f in files:

                    # get file name
                    file_path = os.path.join(root, f)

                    # load audio file
                    signal, sr = librosa.load(file_path)

                    # ensure file is at least 1 sec
                    if len(signal) >= SAMPLES_TO_CONSIDER:
                        # take only first second   
                        signal = signal[:SAMPLES_TO_CONSIDER]

                        # extract mfcc
                        MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                        data["labels"].append(i-1)
                        data["MFCCs"].append(MFCCs.T.tolist())
                        data["files"].append(file_path)
                        print(f"file_path {file_path}, labels {i-1}")
        except Exception as e:
            raise CustomException(e,sys)

    # store in json
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
        logging.info("json_file written to {}".format(json_path))


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)
import keras
from exception import CustomException
from logger import logging
import sys, os
import numpy as np
import librosa


MODEL_PATH = os.path.join('assets', "model.h5")
NUM_SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    model=None
    _mappings=[
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "four",
        "go",
        "happy",
        "house",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "wow",
        "yes",
        "zero"
    ]
    _instance = None

    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)

        # Convert to 4d array
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]


        # Make predictions
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length = 512):
        
        # load the file
        signal, sr = librosa.load(file_path)


        # Consistancey in file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        #Extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T

def Keyword_Spotting_Service():
    # Enforce single instance of KSS

    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    res1 = kss.predict(r"data\data\wow\0a7c2a8d_nohash_1.wav")
    res2 = kss.predict(r"C:\Users\Admin\Downloads\Recording.wav")

    print(f"results are: {res1}, {res2}")
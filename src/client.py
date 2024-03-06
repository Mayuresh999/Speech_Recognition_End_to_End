import os, sys
import requests
from logger import logging
from exception import CustomException


URL = "http://127.0.0.1:5000/predict"

current_directory = os.path.dirname(os.path.abspath(__file__))

TEST_AUDIO_FILE_PATH = os.path.join(current_directory,"test","nine.wav")
# TEST_AUDIO_FILE_PATH = os.path.join("test", "nine.wav")
# TEST_AUDIO_FILE_PATH = "nine.wav"


if __name__ == "__main__":
    try: 
        audio_file = open(TEST_AUDIO_FILE_PATH, 'rb') 
        values = {'file': (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
        response = requests.post(URL, files=values)
        logging.info("Audio file sent to server for prediction")
        data = response.json()
        logging.info(f"Response from server for prediction received. Response is: {data['keyword']}")

        print(f"Predicted Keyword is : {data['keyword']}")
    except Exception as e:
        raise CustomException(e, sys)
from flask import Flask, request, jsonify
import random
from keyword_spotting_service import Keyword_Spotting_Service
import sys
import os
from exception import CustomException
from logger import logging
from waitress import serve

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    # Get the audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0,1000))
    audio_file.save(file_name)
    logging.info("Audio file temporarily saved")

    # invoke the KetwordSpotting service
    kss = Keyword_Spotting_Service()    
    logging.info("Singleton instance of Keyword Spotting Service invoked.")

    # predict 
    predict_response = kss.predict(file_name)
    logging.info("Prediction done")

    # remove the audio file
    os.remove(file_name)
    logging.info("Temporary file removed")

    # send back predictions in json format
    data = {"keyword" : predict_response}
    logging.info("Keyword prediction done and response sent")
    return jsonify(data)

mode = "dev_waitress"

if __name__ == "__main__":
    try:
        if mode != "dev":
            serve(app=app, threads=1, listen = "*:5000", url_scheme='http')
        else:    
            app.run(debug=False)   
    except Exception as e:
        raise CustomException(e,sys) 
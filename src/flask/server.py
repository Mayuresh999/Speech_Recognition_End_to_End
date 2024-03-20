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




# import os
# import requests
# from logger import logging  # Assuming logger is implemented elsewhere
# from exception import CustomException  # Assuming CustomException is defined

# # Replace with your actual server URL
# URL = "http://0.0.0.0/predict"

# # Modify these paths if necessary
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'wav', 'mp3'}  # Add allowed audio extensions

# def allowed_file(filename):
#   return '.' in filename and \
#          filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# from flask import Flask, request, jsonify, render_template
# import random
# from keyword_spotting_service import Keyword_Spotting_Service

# app = Flask(__name__)

# # Configure upload folder (create if it doesn't exist)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(os.path.join(app.instance_path, UPLOAD_FOLDER), exist_ok=True)

# @app.route ('/')
# def home():
#     return render_template('src/flask/templates/index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the uploaded audio file
#     audio_file = request.files.get('audio_file')
#     print(audio_file)

#     if not audio_file:
#         return jsonify({'error': 'No audio file uploaded'}), 400

#     if not allowed_file(audio_file.filename):
#         return jsonify({'error': 'Unsupported file format'}), 415

#     # Generate a random filename to avoid conflicts
#     file_name = f'{random.randint(100000, 999999)}.{audio_file.filename.rsplit(".", 1)[1]}'

#     # Save the uploaded file
#     audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
#     logging.info(f"Audio file saved: {file_name}")

#     # Invoke the Keyword Spotting Service
#     try:
#         kss = Keyword_Spotting_Service()  # Assuming this creates an instance
#         predicted_keyword = kss.predict(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
#         logging.info(f"Prediction completed. Keyword: {predicted_keyword}")
#     except Exception as e:
#         logging.error(f"Error during prediction: {e}")
#         return jsonify({'error': 'Error during prediction'}), 500

#     # Remove the temporary audio file
#     try:
#         os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
#         logging.info(f"Temporary file removed: {file_name}")
#     except FileNotFoundError:
#         logging.warning(f"Temporary file not found: {file_name}")

#     # Send back predictions in JSON format
#     return jsonify({'keyword': predicted_keyword})



# if __name__ == "__main__":
#     app.run(host='http://127.0.0.1/', port='5000', debug=True)
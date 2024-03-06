FROM tensorflow/tensorflow:latest

WORKDIR /app

ADD . /app

# RUN  apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# Install python packages
RUN pip install --upgrade pip setuptools

RUN pip install --no-cache-dir -r requirements.txt --ignore-installed

# Start uWSGI using config file
# CMD ["python", "server.py"]
CMD gunicorn --bind 0.0.0.0:5000 server:app
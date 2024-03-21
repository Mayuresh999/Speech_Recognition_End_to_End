# FROM tensorflow/tensorflow:latest
FROM ubuntu:latest

WORKDIR /app

ADD . /app

RUN  apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

RUN apt-get install -y python3 python3-pip

RUN pip install --upgrade pip setuptools

RUN pip install --no-cache-dir -r requirements.txt --ignore-installed

CMD ["uwsgi", "app.ini"]
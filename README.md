# Speech Recognition End to End 🎙️🔊

## Overview ℹ️

This project is a machine learning-based system that classifies audio signals into various categories. 
It consists of client and server components, leveraging technologies such as Flask, Docker, Keras, TensorFlow, uWSGI, nginx, and Waitress.

## Components 🛠️


![Presentation1](https://github.com/Mayuresh999/Speech_Recognition_End_to_End/assets/95702726/e418f42f-89d4-48a7-95ed-0ae16136f4d1)



### Client 💻

Description: Takes a .wav file and sends it to the Flask server.

Functionality: Initiates communication with the server for audio classification.

### Server 🖥️

Description: Hosts the machine learning model and handles audio file processing and classification.

Components:

#### Nginx Docker Container 🐳: Receives audio files from the system's port and forwards them to the Flask container.
#### Flask Container 🌐: Receives audio files, preprocesses them using keyword_spotting_service, and predicts their category.
#### uWSGI Protocol with Waitress 🚀: Handles communication between the Nginx and Flask containers.
#### keyword_spotting_service 🎛️: Preprocesses audio files and performs predictions using a singleton instance for optimized inference time.
#### Machine Learning Model (CNN with Keras/TensorFlow) 🧠: Used for audio classification.
#### Logging and Exception Handling 📝⚠️

Logging: Implemented to capture logs for each run and facilitate debugging and monitoring.

Custom Exception Handling: Includes a custom exception to manage and handle errors gracefully.

#### Docker Compose 🐋

Description: Orchestrates the deployment and running of Docker containers for seamless execution.

Usage: Simplifies container management and configuration.

### Future Enhancements 🚀✨
Integrate an HTML interface for user file submission.
Support multiple audio file formats and internally convert them to .wav for processing.
Expand compatibility to handle audio files of varying lengths.

### Usage 🚀
Install Docker and Docker Compose.
Clone the repository.
Run docker-compose up to start the containers.
Access the client interface to submit audio files for classification.

### License 📜

This project is licensed under the MIT License - see the LICENSE file for details.

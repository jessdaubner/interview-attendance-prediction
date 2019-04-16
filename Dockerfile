# Use an official Python runtime as a parent image
FROM python:3.7.3

# Copy the current directory contents into the container at /predictor
COPY . /predictor

# Modify enviornment variables for running app in container
ENV PYTHONPATH="${PYTHONPATH}:/predictor"
ENV APP_DIR="predictor"

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r predictor/requirements.txt

# Run program to build model when the container launches
CMD python predictor/preprocess/clean_data.py && python predictor/model/model.py

# Set the working directory to /predictor
WORKDIR predictor/

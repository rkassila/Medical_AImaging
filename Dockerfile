FROM python:3.10.6-buster

WORKDIR /app

COPY models/ /app/models/
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY aimaging/ /app/aimaging/

COPY .env /app/.env

CMD uvicorn aimaging.api.fast:app --host 0.0.0.0 --port 8080

FROM python:3.10.6-buster

WORKDIR /app

COPY models/organ_detection_model.h5 /app/models/organ_detection_model.h5
COPY requirements.txt /app/requirements.txt
COPY aimaging/ /app/aimaging/

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY .env /app/.env

CMD uvicorn aimaging.api.fast:app --host 0.0.0.0 --port 8080

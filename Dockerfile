FROM python:3.10.6-buster

WORKDIR /app

COPY aimaging/ /app/aimaging/
COPY requirements.txt /app/requirements.txt
COPY .env /app/.env
COPY models/organ_detection_model.h5 /app/models/organ_detection_model.h5

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

CMD uvicorn aimaging.api.fast:app --host 0.0.0.0 --port 8000

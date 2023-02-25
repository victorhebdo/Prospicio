FROM python:3.8.12-buster

COPY Prospicio /Prospicio
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn Prospicio.api.fast:app --host 0.0.0.0 --port $PORT

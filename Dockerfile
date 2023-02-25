FROM python:3.8.12-slim-buster

COPY prospicio /prospicio
COPY models /models
COPY requirements_prod.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn prospicio.api.fast:app --host 0.0.0.0 --port $PORT

FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install libgl1 -y

COPY fingerspelling fingerspelling
COPY setup.py setup.py

COPY models models

RUN pip install .

CMD uvicorn fingerspelling.api.api:app --host 0.0.0.0 --port $PORT

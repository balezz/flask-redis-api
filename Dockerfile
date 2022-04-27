FROM python:3.7-buster

RUN apt-get update -y
RUN apt-get install redis-server -y
WORKDIR /usr/src/app

ENV LANG C.UTF-8

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x ./start.sh

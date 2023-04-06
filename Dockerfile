FROM python:3

ENV PYTHONUNBUFFERED 1

WORKDIR /crypto

ADD . /crypto

COPY ./requirements.txt /crypto/requirements.txt

RUN pip install -r requirements.txt

COPY . /crypto

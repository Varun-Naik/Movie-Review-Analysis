# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

# configure the container to run in an executed manner
ENTRYPOINT [ "python3" ]

CMD ["application.py" ]
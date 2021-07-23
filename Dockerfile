FROM ubuntu:20.04

RUN apt update
RUN apt install -y tesseract-ocr poppler-utils libxext-dev libsm-dev libxrender-dev virtualenv python3.8-venv

# create virtual environment and activate it
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY setup.py /app/setup.py
COPY README.md /app/README.md
RUN pip install .

COPY / /app

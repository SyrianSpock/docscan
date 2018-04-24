FROM ubuntu:18.04

RUN apt-get update \
    && apt-get install -y \
        tesseract-ocr-all \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y \
        wget \
        python \
        libsm6 \
        libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py' \
    && python get-pip.py \
    && rm -f get-pip.py

ADD requirements.txt .
RUN pip install -r requirements.txt

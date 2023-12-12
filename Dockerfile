FROM --platform=linux/amd64 python:3.10

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    python3-pip \
    make \
    wget \
    ffmpeg \
    libsm6 \
    libxext6

WORKDIR /service

COPY requirements/ requirements/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements/prod.txt
RUN pip install --no-cache-dir -r requirements/dev.txt

COPY . .

EXPOSE 5000

CMD make start

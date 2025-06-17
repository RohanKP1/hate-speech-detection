FROM python:3.13-slim

WORKDIR /app

# System dependencies for tesseract, ffmpeg, etc.
RUN apt-get update && \
    apt-get install -y build-essential portaudio19-dev libglib2.0-0 libsm6 libxext6 libxrender-dev tesseract-ocr alsa-utils libtesseract-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
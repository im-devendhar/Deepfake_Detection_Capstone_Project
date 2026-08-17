#pull the nvidia cuda GPU docker image
FROM nvidia/cuda

#pull python 3.11 docker image
FROM python:3.11
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
#create a directory to serve static files 
RUN mkdir -p /home/app/staticfiles/app/uploaded_videos/
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install cmake
RUN pip install opencv-python
RUN pip install -r requirements.txt
COPY . /app
RUN chmod +x /app/bin/gunicorn_start.sh
RUN python manage.py collectstatic --noinput
RUN pip install gunicorn
RUN mkdir -p /app/uploaded_videos/app/uploaded_videos/

VOLUME /app/run/
ENTRYPOINT ["/bin/bash", "/app/bin/gunicorn_start.sh"]

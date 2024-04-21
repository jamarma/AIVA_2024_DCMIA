FROM nvidia/cuda:12.1.0-base-ubuntu22.04
LABEL authors="Javier M. Madruga, Hugo G. Parente"

RUN apt-get update && \
    apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6

WORKDIR /dcmia-app

COPY dcmia/ ./dcmia/
RUN pip install -e dcmia/
RUN pip install torch==2.2.1 torchvision==0.17.1

COPY web_app/ ./web_app/
RUN pip install -r web_app/requirements.txt

ENV FLASK_APP="web_app/entrypoint"
ENV APP_SETTINGS_MODULE="config.local"

CMD ["flask", "run", "--host=0.0.0.0"]
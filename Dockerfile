FROM tensorflow/tensorflow:1.15.2-py3

LABEL maintainer="Wim Florijn <wimflorijn@hotmail.com>"

ARG PORT=5000
ARG WORKERS=1
ARG TIMEOUT=60
ARG MAX_REQUESTS=500
ARG WEIGHTS_LOCATION=cooked/vectorisation/model_weights/detector.h5

RUN apt-get update \
 && apt-get install -y libsm6 libxext6 libxrender-dev

COPY docker/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app

COPY wsgi.py /app/wsgi.py
COPY get_weights.py /app/get_weights.py
COPY mrcnn /app/mrcnn


COPY docker/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENV PORT=$PORT
ENV WORKERS=$WORKERS
ENV TIMEOUT=$TIMEOUT
ENV MAX_REQUESTS=$MAX_REQUESTS
ENV WEIGHTS_LOCATION=$WEIGHTS_LOCATION

EXPOSE $PORT

ENTRYPOINT ["/app/entrypoint.sh"]
CMD [ "run" ]

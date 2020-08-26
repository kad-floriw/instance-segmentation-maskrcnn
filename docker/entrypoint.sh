#!/bin/bash
set -e

if [ "$1" = "run" ]; then
  WSGI_MODULE=wsgi
  WORKER_CLASS=gevent
  WORKER_CONNECTIONS=10
  MAX_REQUESTS_JITTER=10

  python get_weights.py

  exec gunicorn ${WSGI_MODULE}:app \
    --worker-class=$WORKER_CLASS \
    --worker-connections=$WORKER_CONNECTIONS \
    --workers "$WORKERS" \
    --timeout "$TIMEOUT" \
    --max-requests "$MAX_REQUESTS" \
    --max-requests-jitter $MAX_REQUESTS_JITTER \
    --bind=0.0.0.0:"$PORT"
else
  exec "$@"
fi

ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements2.txt,target=requirements2.txt \
    python -m pip install -r requirements2.txt

RUN pip install --no-deps --no-cache-dir crawl4ai

COPY . .

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

CMD ["streamlit", "run", "main.py", "--server.port=8080"]
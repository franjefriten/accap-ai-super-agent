FROM selenium/standalone-chrome:latest AS selenium-base

FROM python:3.12-slim AS base

COPY --from=selenium-base /usr/bin/google-chrome /usr/bin/google-chrome
COPY --from=selenium-base /usr/bin/chromedriver /usr/bin/chromedriver

ENV PATH="/opt/chromedriver:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    gnupg \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libnss3 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements2.txt .
RUN pip install --no-cache-dir -r requirements2.txt
RUN pip install --no-cache-dir crawl4ai==0.5.0.post8

COPY . .

EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health
CMD ["streamlit", "run", "main.py", "--server.port=8080"]
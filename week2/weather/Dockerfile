FROM python:3.14-slim

WORKDIR /app

# Avoid writing .pyc files and force unbuffered logging for cleaner container logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies first to make rebuilding lightning-fast via layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY README.md README.ru.md ./

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

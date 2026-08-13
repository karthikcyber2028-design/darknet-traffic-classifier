FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    FLASK_DEBUG=0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# trained model artifacts are gitignored; generate them during build
RUN python main.py compare --quick

EXPOSE 8000
CMD ["gunicorn", "webapp.app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]

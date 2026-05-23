FROM python:3.11-slim

# Install system libs needed by Pillow and PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Optional but recommended: use gunicorn instead of Flask dev server
# Add "gunicorn" to your requirements.txt if you use the CMD below

COPY . .

EXPOSE 8000

# Production CMD (add gunicorn to requirements.txt):
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "120", "main:app"]

# OR keep Flask dev server (slower but works):
# CMD ["python", "main.py"]

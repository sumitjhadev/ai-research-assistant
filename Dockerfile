# AI Research Assistant — backend API image
#
# This image runs the FastAPI backend (uvicorn). The Streamlit frontend is
# intended to be deployed separately (Streamlit Community Cloud) and talk to
# this backend over HTTP via the BACKEND_URL environment variable, but the
# same image can also run the Streamlit app locally via docker-compose.
FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss-cpu / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]

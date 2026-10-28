FROM python:3.12-slim

# torch / sentence-transformers wheels are mature on 3.12; the app itself is
# version-agnostic. CPU wheels keep the image out of the multi-GB range.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY scripts ./scripts

RUN mkdir -p /app/data/pdfs /app/data/chroma_db

EXPOSE 8000
CMD ["uvicorn", "tiered_rag.service:app", "--host", "0.0.0.0", "--port", "8000"]

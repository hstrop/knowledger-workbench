FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN python -m pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src \
    KNOWLEDGER_MODE=offline \
    KNOWLEDGER_HOST=0.0.0.0 \
    KNOWLEDGER_PORT=8000 \
    KNOWLEDGER_WORKSPACE=/app/data
RUN mkdir -p /app/data
VOLUME ["/app/data"]
EXPOSE 8000
CMD ["uvicorn", "knowledger.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Isolated code-execution service for the Ops Analyst: no internet, read-only FS, CPU/memory limits (see compose).
FROM python:3.12-slim
RUN pip install --no-cache-dir pandas==3.0.6 numpy==2.5.2 fastapi==0.162.0 uvicorn==0.54.0
RUN useradd -m runner
WORKDIR /srv
COPY sandbox/server.py backend/lifeline/tools/sandbox.py ./
USER runner
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]

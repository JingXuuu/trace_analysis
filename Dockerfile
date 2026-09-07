FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-work \
    MPLCONFIGDIR=/tmp/matplotlib-cache

RUN apt-get update \
    && apt-get install --no-install-recommends -y graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir '.[deployment]'

COPY traces ./traces
ARG TRACE_ANALYSIS_UID=10001
ARG TRACE_ANALYSIS_GID=10001
RUN groupadd --gid "${TRACE_ANALYSIS_GID}" trace-analysis \
    && useradd --create-home --uid "${TRACE_ANALYSIS_UID}" \
        --gid "${TRACE_ANALYSIS_GID}" trace-analysis \
    && mkdir -p /data/traces /data/artifacts /tmp/flashinfer-work /tmp/matplotlib-cache \
    && chown -R trace-analysis:trace-analysis /data /tmp/flashinfer-work /tmp/matplotlib-cache

USER trace-analysis
EXPOSE 8080
VOLUME ["/data/traces", "/data/artifacts"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=3)"]

CMD ["trace-radix-web", "--host", "0.0.0.0", "--port", "8080", "--trace-dir", "/app/traces", "--trace-dir", "/data/traces", "--output-dir", "/data/artifacts"]

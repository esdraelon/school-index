# School Index MCP — production image (M4 deploy artifact).
#
# Build from the repo root:
#   docker build -f deploy/Dockerfile -t haliffer-index-mcp .
#
# Required environment:
#   INDEX_MCP_TOKEN  the MCP's OWN bearer <redacted> the gateway presents
#                    (registered path-scoped as client "index-mcp")
#   CORPUS_ROOT      corpus root inside the data mount (/data)
# Optional: INDEX_MCP_HOST, INDEX_MCP_PORT, MODELS_DIR (defaults to
# <CORPUS_ROOT>/models), EMBED_MODEL (must match the index's model).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INDEX_MCP_HOST=0.0.0.0 \
    INDEX_MCP_PORT=8766 \
    CORPUS_ROOT=/data

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install . \
    && useradd --create-home --uid 10001 index

USER index
EXPOSE 8766

CMD ["python", "-m", "school_index.index_mcp"]

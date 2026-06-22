FROM python:3.11-slim

ARG INSTALL_CLOUD_DEPS=false

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

RUN if [ "$INSTALL_CLOUD_DEPS" = "true" ]; then \
        uv add boto3; \
    fi

COPY src/ /app/src/
RUN mkdir -p /app/export

CMD ["uv", "run", "python", "src/flow.py"]
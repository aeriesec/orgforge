# ── OrgForge — App Container ─────────────────────────────────
# Runs flow.py (simulation) and email_gen.py (emails).
#
# Build args:
#   INSTALL_CLOUD_DEPS=true   installs boto3, langchain-aws, openai
#                             required for quality_preset: "cloud"
#                             default: false (keeps image lean for local presets)
#
# Examples:
#   Local preset:  docker build -t orgforge .
#   Cloud preset:  docker build --build-arg INSTALL_CLOUD_DEPS=true -t orgforge .
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

ARG INSTALL_CLOUD_DEPS=false

# ── System deps ───────────────────────────────────────────────
# curl:        Ollama connection probing
# gcc + libffi-dev: required by crewai/pydantic native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY requirements-cloud.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN if [ "$INSTALL_CLOUD_DEPS" = "true" ]; then \
        pip install --no-cache-dir -r requirements-cloud.txt; \
    fi

# ── Application code ──────────────────────────────────────────
# Copy only source files — not export, .env, or local config overrides.
# config.yaml is bind-mounted at runtime (see docker-compose.yaml) so edits
# on the host are picked up without rebuilding the image.
COPY src/ /app/src/

# Output directory — created here so the bind mount has a target
# even if ./export doesn't exist on the host yet.
RUN mkdir -p /app/export

# ── Default command ───────────────────────────────────────────
# Run flow.py from its new home in the src directory
CMD ["python", "src/flow.py"]

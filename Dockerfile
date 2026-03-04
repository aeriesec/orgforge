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

# ── Core Python dependencies ──────────────────────────────────
# Installed explicitly rather than via -r requirements.txt so commented-out
# cloud dep blocks in requirements.txt don't cause parse errors.
RUN pip install --no-cache-dir \
    "crewai>=0.28.0" \
    "crewai-tools>=0.1.0" \
    "langchain-community>=0.0.20" \
    "pymongo>=4.6.0" \
    "requests>=2.31.0" \
    "pyyaml>=6.0" \
    "pydantic>=2.0.0" \
    "networkx>=3.0" \
    "vaderSentiment>=3.3.2" \
    "rich>=13.0.0" \
    "ollama>=0.1.7"

# ── Optional: cloud preset deps ───────────────────────────────
# Only installed when INSTALL_CLOUD_DEPS=true.
# Not needed for local_cpu or local_gpu presets.
RUN if [ "$INSTALL_CLOUD_DEPS" = "true" ]; then \
        pip install --no-cache-dir \
            "boto3>=1.34.0" \
            "langchain-aws>=0.1.0" \
            "openai>=1.12.0"; \
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

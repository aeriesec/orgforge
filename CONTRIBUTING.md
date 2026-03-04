# Contributing to OrgForge

First off, thank you for considering contributing to OrgForge! Our goal is to build the definitive open-source synthetic dataset generator for testing enterprise AI agents.

## 🤝 How to Contribute

We welcome all types of contributions, but here is what we are looking for most right now:

1. **New Domain Configs:** Well-tuned `config.yaml` templates for specific industries (e.g., healthcare, legal, fintech).
2. **New Artifact Types:** Handlers for Zoom transcripts, Zendesk tickets, or PagerDuty alerts.
3. **Bug Fixes:** Improvements to the state machine, memory retrieval, or edge cases.

### The Golden Rule

**Before you write thousands of lines of code, please open an Issue.** We want to make sure your work aligns with the project roadmap so we don't have to reject your Pull Request.

## 🛠️ Local Development Setup

To develop locally, you'll need Python 3.11+, Docker, and ideally a local instance of Ollama.

1. **Fork and clone the repository:**

```bash
git clone [https://github.com/YOUR_USERNAME/orgforge.git](https://github.com/YOUR_USERNAME/orgforge.git)
cd orgforge
```

2. **Set up a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

3. **Install dependencies (including testing tools):**

```bash
pip install -r requirements-test.txt

```

4. **Set up your environment variables:**
   Copy the example file and fill it out if you plan to test cloud LLMs.

```bash
cp .env.example .env

```

## 🧪 Testing

OrgForge relies heavily on a strict test suite that heavily mocks LLM and Database calls to ensure the underlying logic holds up.

**You must ensure all tests pass before submitting a Pull Request.**

To run the test suite, ensure your `PYTHONPATH` points to the `src` directory:

```bash
# On Linux / macOS
PYTHONPATH=src pytest tests/ -v

# On Windows (PowerShell)
$env:PYTHONPATH="src"; pytest tests/ -v

```

## 🏗️ Adding a New Artifact Type

The codebase is structured for extension. If you are adding a new artifact type (like a Zendesk ticket simulator):

1. Add an event emission (`self._mem.log_event(...)`) in `src/flow.py` when the triggering condition occurs.
2. Add the artifact logic to handle the LLM generation.
3. Ensure the artifact grounds its facts using the `SimEvent` context.
4. Write tests for your new logic in the `tests/` directory.

## 📝 Pull Request Process

1. Create a new branch for your feature (`git checkout -b feature/amazing-new-thing`).
2. Make your changes and commit them with descriptive messages.
3. Run the tests (`PYTHONPATH=src pytest tests/`).
4. Push your branch to your fork and open a Pull Request against the `main` branch.
5. A maintainer will review your code, potentially ask for changes, and merge it!

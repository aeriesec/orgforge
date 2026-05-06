# VeloMind Viewer Setup

## 1. Place the Export

From the repo root, the folder should look like this:

```text
orgforge/
  export/
    velomind_30d/
      simulation_snapshot.json
      emails/
      slack/
      jira/
      confluence/
      ...
```

If the zip extracts to just `velomind_30d`, move it under `export/`.

## 2. Run One Command

```bash
uv run --no-project --with-requirements requirements.txt python scripts/view_export.py export/velomind_30d
```

Open:

```text
http://localhost:8765
```

That command does:

- starts the repo's MongoDB container if MongoDB is not already running
- imports `export/velomind_30d/simulation_snapshot.json` into `orgforge_velomind_30d`
- validates the imported dataset
- launches the local viewer on port `8765`

## Without `uv`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/view_export.py export/velomind_30d
```

## Useful Options

Use a different port:

```bash
uv run --no-project --with-requirements requirements.txt python scripts/view_export.py export/velomind_30d --port 8770
```

Import and validate, but do not launch the viewer:

```bash
uv run --no-project --with-requirements requirements.txt python scripts/view_export.py export/velomind_30d --import-only
```

Launch against an already-imported DB without re-importing:

```bash
uv run --no-project --with-requirements requirements.txt python scripts/view_export.py export/velomind_30d --skip-import
```

Use an existing MongoDB instead of starting Docker:

```bash
uv run --no-project --with-requirements requirements.txt python scripts/view_export.py export/velomind_30d \
  --no-docker \
  --mongo-uri "mongodb://localhost:27017/?directConnection=true"
```

## Troubleshooting

- **Docker is not installed:** start MongoDB another way and run with `--no-docker --mongo-uri ...`.
- **Port already in use:** rerun with `--port 8770`.
- **Empty viewer:** make sure the export path is exactly `export/velomind_30d` and rerun without `--skip-import`.
- **Validation fails:** confirm the zip includes `simulation_snapshot.json` and was extracted without nesting an extra folder.

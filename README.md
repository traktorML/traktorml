# TraktorML

A TUI (Text User Interface) on top of an MLFlow Tracking Server.

## Dev Setup

### Install project dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Install pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files --hook-stage manual
```

### Setup .env file

You can run the TUI with a local MLFlow Tracking server or with a remote server. If you dont set the
environment variable `MLFLOW_TRACKING_URI`, the TUI will use a local MLFlow Tracking server (file
based in the folder `mlruns`). The TUI will read and set environment variables from the `.env` file
(see `.env.template` for a template).

### Generate fake data

If you dont have any data in your MLFlow Tracking server (local or remote), you can generate some
fake data with the following command

```bash
python generate_fake_data.py
```

### Run the TUI

```bash
python traktorml.py
```

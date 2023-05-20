# TraktorML


## Dev Setup

### Install project dependencies
```bash
pip install -r requirements.txt
```

### Setup .env file
If you are using a non local (file based) MLFlow Tracking server, you need to set the `MLFLOW_TRACKING_URI` environment variable.
Copy the `.env.template` file to `.env` and fill in the values.

### Run the TUI
```bash
python traktorml.py
```

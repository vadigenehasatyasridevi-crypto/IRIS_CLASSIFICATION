# Iris Classification

A Flask app that predicts an Iris species from sepal and petal measurements.

## Setup

From this project folder, install the dependencies:

```powershell
python -m pip install -r ".\MODEL\requirements.txt"
```

## Run the app

The trained model is stored in `MODEL\iris_model.pkl`. Start the Flask app with:

```powershell
cd ".\MODEL"
python app.py
```

Then open <http://127.0.0.1:5000> in a browser.

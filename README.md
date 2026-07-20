# Breast Cancer Diagnosis Predictor 🩺

This is a small end-to-end machine learning project I built to practice
taking a model from raw data all the way to something usable — training,
an API, and a simple UI to try it out.

It predicts whether a tumor is **benign** or **malignant** based on 30
measurements taken from cell samples (things like radius, texture, and
smoothness). The dataset is the well-known Breast Cancer Wisconsin dataset,
and it comes bundled with scikit-learn, so there's nothing to download.

Everything here is free and open source. No API keys, no paid services,
no sign-ups — just clone it and run it.

## What's in this project

- **`src/train.py`** – loads the data, trains a Random Forest model, and
  saves it
- **`src/predict.py`** – loads the saved model and makes predictions
- **`api/main.py`** – a small FastAPI app that serves predictions over a
  REST API
- **`demo_app.py`** – a Streamlit app with a simple form, so you can test
  predictions without writing any code
- **`tests/`** – some basic tests to make sure everything works

## How to run it

```bash
# clone the repo
git clone https://github.com/<your-username>/ml-diagnosis-pipeline.git
cd ml-diagnosis-pipeline

# install the dependencies
pip install -r requirements.txt

# train the model (takes a few seconds)
python src/train.py
```

Then pick one of these:

**Try the API:**
```bash
uvicorn api.main:app --reload
```
Open `http://127.0.0.1:8000/docs` in your browser — it gives you a page
where you can test predictions right from the browser.

**Or try the interactive demo instead:**
```bash
streamlit run demo_app.py
```
This opens a form where you can adjust values and see the prediction
update live.

## How the model works

I used a Random Forest classifier with a `StandardScaler` for
preprocessing, and tuned it with a small grid search to find the best
settings. On the test set it gets around **96% accuracy**.

## Running the tests

```bash
pytest -v
```

## A note

This is a learning/portfolio project, not a real diagnostic tool —
please don't use it to make actual medical decisions.

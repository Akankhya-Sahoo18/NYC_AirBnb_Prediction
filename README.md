# StaySense — NYC Room Type Classifier

A Streamlit dashboard for your NYC Airbnb room-type classification model.

## Features
- Predicts **Entire home/apt**, **Private room**, or **Shared room**
- Uses the saved `Model_Pipeline.pkl` model pipeline
- Shows per-class model probabilities
- Explores room-type distribution, nightly prices, borough mix, and sample rows
- Includes model features, preprocessing details, and evaluation scores recorded in the notebook

## Project structure

```text
NYC_Room_Type_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── Model_Pipeline.pkl       # copy your existing file here
└── data/
    └── AB_NYC_2019.csv
```

## Setup on Windows (recommended: Python 3.11)

Open this folder in VS Code. Make sure your existing `Model_Pipeline.pkl` is copied into the project root beside `app.py`.

Open the VS Code terminal and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run the equivalent Command Prompt activation command:

```bat
.venv\Scripts\activate.bat
```

Then run:

```bat
streamlit run app.py
```

Streamlit should open the app in your browser, usually at `http://localhost:8501`.

## Important model compatibility note

The uploaded pipeline was serialized with **scikit-learn 1.4.2**. Use the version pinned in `requirements.txt` to reduce model-loading compatibility problems. Python 3.11 is recommended for this environment. If you use a different Python version or have an existing environment with another scikit-learn version, create the virtual environment above rather than downgrading packages in your global Python installation.

## Notes
- The app expects `Model_Pipeline.pkl` beside `app.py`.
- The CSV is used for dashboard charts and neighborhood defaults. Prediction requires the model file.
- The UI uses the same 10 feature names and ordering represented in your saved pipeline, passing a DataFrame with named columns.
- The notebook capped `price` at its 99th percentile (~$799) and `minimum_nights` at 45 before training; the UI input ranges reflect those caps.
- The notebook's recorded evaluation metrics are displayed as reported values, not recalculated by the app.
- Only load pickle/joblib model files you trust, because these formats can execute code when loaded.

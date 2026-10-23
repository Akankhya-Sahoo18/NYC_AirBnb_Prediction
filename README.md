# NYC Airbnb Room Type Prediction

A machine learning web application that predicts the room type of an Airbnb listing in New York City using listing details such as location, price, minimum nights, reviews, and availability.

The project uses a **Random Forest classifier** wrapped in a Scikit-learn preprocessing pipeline and a **Streamlit** interface to make predictions interactively.

## Project Overview

Airbnb listings vary by neighbourhood, price, availability, and other characteristics. This project explores these listing attributes to build a multiclass classification model that predicts one of three room types:

- Entire home/apt
- Private room
- Shared room

Users can enter listing details through the web interface and receive a predicted room type.

## Tech Stack

- **Python** — data processing and model development
- **Pandas & NumPy** — data manipulation
- **Scikit-learn** — preprocessing, model training, and prediction
- **Random Forest Classifier** — classification algorithm
- **Streamlit** — interactive web application
- **Joblib** — saving and loading the trained model

## Dataset

The project uses the **New York City Airbnb Open Data (2019)** dataset, `AB_NYC_2019.csv`.

The dataset contains Airbnb listing information, including neighbourhoods, prices, reviews, availability, and room types.

## Features Used

The model uses the following input features:

- Neighbourhood group
- Neighbourhood
- Latitude and longitude
- Price
- Minimum nights
- Number of reviews
- Reviews per month
- Calculated host listing count
- Availability throughout the year

The target variable is `room_type`.

## Machine Learning Workflow

1. **Data preparation:** Selected relevant features and the target variable.
2. **Data cleaning:** Handled missing values and limited extreme values in selected numerical features.
3. **Preprocessing:** Used median imputation and standard scaling for numerical features, and most-frequent imputation and one-hot encoding for categorical features.
4. **Model training:** Trained a Random Forest classifier with class weighting.
5. **Model evaluation:** Evaluated classification performance using accuracy, macro F1-score, and cross-validation.
6. **Model persistence:** Saved the trained preprocessing and prediction pipeline as `Model_Pipeline.pkl`.
7. **Deployment interface:** Integrated the saved pipeline into a Streamlit application for interactive predictions.

## Project Structure

```text
NYC_Room_Predictions/
├── app.py
├── Model_Pipeline.pkl
├── requirements.txt
├── README.md
└── data/
    └── AB_NYC_2019.csv
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Akankhya-Sahoo18/NYC_AirBnb_Prediction.git
cd NYC_AirBnb_Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Model Performance

The model achieved the following results during evaluation:

| Metric | Score |
|---|---:|
| Accuracy | 85.66% |
| Macro F1-score | 74.06% |
| Best cross-validation macro F1-score | 73.32% |

Accuracy measures overall correct predictions, while macro F1-score gives equal weight to each room type. Using both provides a more informative evaluation when classes are imbalanced.


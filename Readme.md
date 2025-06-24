# Credit Risk Prediction App

A user-friendly Streamlit web app that predicts the **risk of loan default** and classifies **customer credit scores** using a trained XGBoost model.

## Features

- Interactive input form for loan applicants
- Predicts loan default probability
- Categorizes applicant based on credit score:
  - Excellent
  - Very Good
  - Good
  - Fair
  - Poor
- Simplified risk messaging and scoring output
- Built using a dataset of over 260,000 records

## Technologies Used

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas, NumPy

## Files in this Repo

- `app.py`: Streamlit app file (main deployment script)
- `model.pkl`: Trained XGBoost model
- `requirements.txt`: Dependencies for Streamlit Cloud
- `Credit Risk Prediction.ipynb`: Jupyter notebook used to train the model
- Dataset used for training

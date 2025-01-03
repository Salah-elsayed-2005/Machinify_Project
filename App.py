import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.base import BaseEstimator

# Define ClfSwitcher class
class ClfSwitcher(BaseEstimator):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if hasattr(self.estimator, 'predict_proba'):
            return self.estimator.predict_proba(X)
        else:
            raise AttributeError(f"The estimator {self.estimator} does not support predict_proba.")

# Load the model
try:
    model = joblib.load("D:/drive/one drive/OneDrive - Egypt University of Informatics/Desktop/Loan_prediction_model_Data1.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please upload or define the model first.")

# Define preprocessing and prediction function
def preprocess_and_predict(data):
    # Feature engineering
    data = pd.concat([data, pd.get_dummies(data['person_home_ownership'], prefix='person_home_ownership')], axis=1)
    data = pd.concat([data, pd.get_dummies(data['loan_intent'], prefix='loan_intent')], axis=1)
    data['cb_person_default_on_file'] = data['cb_person_default_on_file'].replace({'N': 0, 'Y': 1})
    label_encoder = LabelEncoder()
    data['loan_grade'] = label_encoder.fit_transform(data['loan_grade'])

    # Select relevant features for prediction
    columns = [
        'person_age', 'person_income', 'person_emp_length', 'loan_grade', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
        'cb_person_cred_hist_length', 'person_home_ownership_MORTGAGE',
        'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
        'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
        'loan_intent_PERSONAL', 'loan_intent_VENTURE'
    ]
    for column in columns:
        if column not in data.columns:
            data[column] = 0  # Set missing column to zero

    features = data[columns]
    features_Numeric = features[
        ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
         'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    ]

    # Scaling
    scaler_normalizer = Normalizer()
    features_Normalized_NotEncoded = scaler_normalizer.fit_transform(features_Numeric)
    features_Normalized_NotEncoded = pd.DataFrame(features_Normalized_NotEncoded, columns=features_Numeric.columns)
    features.reset_index(drop=True, inplace=True)
    features_Normalized_NotEncoded.reset_index(drop=True, inplace=True)

    scaled_features = pd.concat(
        [features_Normalized_NotEncoded, features[['loan_grade', 'cb_person_default_on_file',
                                                    'person_home_ownership_MORTGAGE', 'person_home_ownership_OWN',
                                                    'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
                                                    'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
                                                    'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE']]],
        axis=1)

    # Predict loan status
    predictions = model.predict(scaled_features)
    data['loan_status'] = predictions
    return data

# App UI design
st.title("💳 Loan Approval Predictor")
st.markdown("""
Welcome to the **Loan Approval Predictor App**!  
Use the sidebar to input your loan application details, and the app will predict whether your loan is likely to be approved.
""")
st.sidebar.title("🔧 Input Loan Data")

# User inputs
with st.sidebar.form("input_form"):
    person_age = st.slider("Age", 18, 75, 30)
    person_income = st.number_input("Annual Income ($)", min_value=0, step=500)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
    person_emp_length = st.number_input("Employment Length (Years)", min_value=0, step=2)
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, step=500)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.01)
    cb_person_default_on_file = st.selectbox("Default on File", ["Y", "N"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, step=1)
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    input_df = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })
    result_df = preprocess_and_predict(input_df)
    st.markdown("### Prediction Results:")
    loan_status = result_df['loan_status'].values[0]
    if loan_status == 1:
        st.success("🎉 Congratulations! Your loan is likely to be approved.")
    else:
        st.error("🚫 Sorry, your loan is not likely to be approved.")

    st.markdown("#### Detailed Output:")
    st.dataframe(result_df)

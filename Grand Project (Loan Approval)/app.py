
# RUN THIS APP BY RUNNING THIS COMMAND ON TERMINAL => streamlit run app.py


import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Load the model
loan_approval_model = joblib.load(open('model.pkl', 'rb'))

categorical_mappings = {
    'home_n': {'Mortgage': 0, 'Other': 1, 'Own': 2,  'Rent': 3},
    'intent_n': {'Debt Consolidation': 0, 'Education': 1, 'Home Improvement': 2, 'Medical Expenses': 3,  'Personal': 4, 'Venture': 5},
    'default_n': {'Y': 1, 'N': 0},  # Add more categories if needed
}

### MAIN FUNCTION ###
def main(title="Loan Approval Prediction App".upper()):
    st.markdown("<h1 style='text-align: center; font-size: 25px; color: white;'>{}</h1>".format(title),
                unsafe_allow_html=True)
    info = ''

    with st.expander("1. Check if your loan application is approved or unapproved"):
        # Collect user input for loan application
        age = st.slider("Select Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Enter Income", min_value=0, value=50000)
        home_status = st.selectbox("Select Home Ownership Status", [
            "Mortgage", "Other", "Own",  "Rent"])
        emp_length = st.slider(
            "Select Employment Length (in years)", min_value=0, max_value=50, value=5)
        intent = st.selectbox(
            "Select Purpose of the loan", ['Debt Consolidation', 'Home Improvement', 'Medical Expenses', 'Venture', 'Education', 'Personal'])
        amount = st.number_input(
            "Enter Loan Amount Applied For")
        rate = st.number_input(
            "Enter Interest Rate on the loan")
        percent_income = st.slider(
            "Select Loan Amount as a Percentage of Income", min_value=0, max_value=100, value=20)
        default_history = st.selectbox("Select Default History", ["Y", "N"])
        cred_length = st.slider(
            "Select Length of Credit History", min_value=0, max_value=30, value=10)

        if st.button("Predict Loan Approval"):

            user_data = pd.DataFrame({
                'Age': [age],
                'Income': [income],
                'home_n': [home_status],
                'Emp_length': [emp_length],
                'intent_n': [intent],
                'Amount': [amount],
                'Rate': [rate],
                'Percent_income': [percent_income],
                'default_n': [default_history],
                'Cred_length': [cred_length]
            })

            # Label encoding for categorical variables
            for column, mapping in categorical_mappings.items():
                le = LabelEncoder()
                user_data[column] = le.fit_transform(
                    user_data[column].map(mapping))
                # Make prediction
            prediction = loan_approval_model.predict(user_data)

            # Display prediction result
            if prediction[0] == 0:
                info = 'UNAPPROVED'
                st.error(f'Prediction: Loan {info}')
            else:
                info = 'APPROVED'
                st.success(f'Prediction: Loan {info}')


if __name__ == "__main__":
    main()

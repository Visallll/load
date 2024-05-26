import streamlit as st
import joblib
import pandas as pd

# Function to load a model
@st.cache(allow_output_mutation=True)
def load_model(uploaded_file):
    return joblib.load(uploaded_file)

# Streamlit app title
st.title('Machine Learning Model Deployment with Streamlit')

# Sidebar for model upload
st.sidebar.header('Model Upload')
uploaded_file = st.sidebar.file_uploader("Upload your model file", type=["pkl"])

if uploaded_file is not None:
    model = load_model(uploaded_file)
    st.sidebar.write("Model loaded successfully!")

    # User inputs
    st.header('Input Parameters')
    def user_input_features():
        feature1 = st.number_input('Feature 1', min_value=0, max_value=100, value=50)
        feature2 = st.number_input('Feature 2', min_value=0, max_value=100, value=50)
        # Add other features as needed
        data = {'feature1': feature1,
                'feature2': feature2}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Display input features
    st.subheader('User Input features')
    st.write(input_df)

    # Model prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    st.write(prediction)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
else:
    st.sidebar.write("Please upload a model file.")

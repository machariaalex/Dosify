import streamlit as st
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Assuming you have the balanced_train and test_data available
# If not, make sure to load your datasets appropriately
balanced_train = pd.read_csv('balld.csv').set_index('SN')
test_data = pd.read_csv('tesiit_data.csv').set_index('SN')

query_columns = pd.read_csv('buttons.csv')

st.image('sanku_logo.png', width=200)

# Create a list of models to fit
models = [BaggingClassifier(),  GradientBoostingClassifier(), RandomForestClassifier()]

# Fit each model to the transformed dataset
for model in models:
    model.fit(balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'])

# Streamlit App
st.title("Dosifier Project Deployment")

# Dropdowns for query columns
st.sidebar.header('Select Columns')
selected_date_added = st.sidebar.selectbox("Select Date ADDED", query_columns['DATE ADDED'].unique())
selected_sn = st.sidebar.selectbox("Select Serial Number", query_columns['SN'].unique())
selected_region = st.sidebar.selectbox("Select REGION", query_columns['REGION'].unique())

# User input for general features
st.sidebar.header('Input Features')
user_input_general = {}
for feature in balanced_train.drop(['CATEGORY'], axis=1).columns:
    user_input_general[feature] = st.sidebar.slider(f'Select {feature}', float(balanced_train[feature].min()), float(balanced_train[feature].max()))

# User input for query columns
user_input_query = {
    'DATE ADDED': selected_date_added,
    'SN': selected_sn,
    'REGION': selected_region
}

# Dropdown to select the model
st.sidebar.header('Select Model')
selected_model_name = st.sidebar.selectbox("Select a model", [model.__class__.__name__ for model in models])
selected_model = next((model for model in models if model.__class__.__name__ == selected_model_name), None)

if selected_model is None:
    st.warning("Invalid model selected.")
    st.stop()


# Create dataframes with user input
user_input_df_query = pd.DataFrame([user_input_query])
user_input_df_general = pd.DataFrame([user_input_general])


# Display user input
st.write("User Input for Query Columns:")
st.write(f"Selected Date: {selected_date_added}")
st.write(f"Selected Serial Number: {selected_sn}")
st.write(f"Selected REGION: {selected_region}")

st.write("User Input:")
st.write(user_input_df_general)

# Determine CATEGORY based on selected query columns
selected_row = query_columns[
    (query_columns['DATE ADDED'] == selected_date_added) &
    (query_columns['SN'] == selected_sn) &
    (query_columns['REGION'] == selected_region)
]

if not selected_row.empty:
    predicted_category_query = selected_row['CATEGORY'].values[0]
    st.write(f"Category for Query Columns: {predicted_category_query}")
else:
    st.warning("No matching row found for the selected columns.")


# Predict the category using the selected model and general features
prediction_general = selected_model.predict(user_input_df_general)
st.write(f"Predicted Category For General Features: {prediction_general[0]}")

# Display model evaluation results
st.header("Model Evaluation Results")
for model in models:
    accuracy = cross_val_score(model, balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'], cv=5)
    st.write(f"Accuracy of {model.__class__.__name__}: {accuracy.mean()}")



# Display predictions and probabilities
predictions = selected_model.predict(test_data.drop(['CATEGORY'], axis=1))
probabilities = selected_model.predict_proba(test_data.drop(['CATEGORY'], axis=1))
dosifier_predictions = pd.DataFrame(probabilities, columns=selected_model.classes_, index=test_data.index)
dosifier_predictions_final = dosifier_predictions.groupby(level=0).mean()
st.header("Test Data Predictions and Probabilities")
st.write(dosifier_predictions_final)
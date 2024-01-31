import streamlit as st
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Assuming you have the balanced_train and test_data available
# If not, make sure to load your datasets appropriately
balanced_train = pd.read_csv('balanced_train.csv')
test_data = pd.read_csv('test_data.csv')

query_columns = pd.read_csv('buttons.csv')

# Create a list of models to fit
models = [BaggingClassifier(), AdaBoostClassifier(algorithm='SAMME'), GradientBoostingClassifier(), RandomForestClassifier()]

# Fit each model to the transformed dataset
for model in models:
    model.fit(balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'])

# Streamlit App
st.title("Dosifier Project Deployment")

# User input for general features
st.sidebar.header('Input Features')
user_input_general = {}
for feature in balanced_train.drop(['CATEGORY'], axis=1).columns:
    user_input_general[feature] = st.sidebar.slider(f'Select {feature}', float(balanced_train[feature].min()), float(balanced_train[feature].max()))

# Dropdowns for query columns
st.sidebar.header('Select Columns')
selected_date_added = st.sidebar.selectbox("Select Date ADDED", query_columns['DATE ADDED'].unique())
selected_sn = st.sidebar.selectbox("Select Serial Number", query_columns['SN'].unique())
selected_region = st.sidebar.selectbox("Select REGION", query_columns['REGION'].unique())

# User input for query columns
user_input_query = {
    'DATE ADDED': selected_date_added,
    'SN': selected_sn,
    'REGION': selected_region
}

# Create dataframes with user input
user_input_df_general = pd.DataFrame([user_input_general])
user_input_df_query = pd.DataFrame([user_input_query])

# Display user input
st.write("User Input (General Features):")
st.write(user_input_df_general)

st.write("User Input (Query Columns):")
st.write(f"Selected Date Added: {selected_date_added}")
st.write(f"Selected Serial Number: {selected_sn}")
st.write(f"Selected REGION: {selected_region}")

# Predict the category using the selected model and general features
selected_model_index = st.sidebar.selectbox("Select a model", range(len(models)))
selected_model = models[selected_model_index]

prediction_general = selected_model.predict(user_input_df_general)
st.write(f"Predicted Category (General Features): {prediction_general[0]}")

# Display model evaluation results
st.header("Model Evaluation Results")
for model in models:
    accuracy = cross_val_score(model, balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'], cv=5)
    st.write(f"Accuracy of {model.__class__.__name__}: {accuracy.mean()}")

# Determine CATEGORY based on selected query columns
selected_row = query_columns[
    (query_columns['DATE ADDED'] == selected_date_added) &
    (query_columns['SN'] == selected_sn) &
    (query_columns['REGION'] == selected_region)
]

if not selected_row.empty:
    predicted_category_query = selected_row['CATEGORY'].values[0]
    st.write(f"Predicted Category (Query Columns): {predicted_category_query}")
else:
    st.warning("No matching row found for the selected columns.")

# Display predictions and probabilities
predictions = selected_model.predict(test_data.drop(['CATEGORY'], axis=1))
probabilities = selected_model.predict_proba(test_data.drop(['CATEGORY'], axis=1))
dosifier_predictions = pd.DataFrame(probabilities, columns=selected_model.classes_, index=test_data.index)
dosifier_predictions_final = dosifier_predictions.groupby(level=0).mean()
st.header("Test Data Predictions and Probabilities")
st.write(dosifier_predictions_final)

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

# User input for features
st.sidebar.header('Input Features')
user_input = {}
for feature in balanced_train.drop(['CATEGORY'], axis=1).columns:
    user_input[feature] = st.sidebar.slider(f'Select {feature}', float(balanced_train[feature].min()), float(balanced_train[feature].max()))

# Dropdowns for additional columns
st.sidebar.header('Select Columns')
selected_date_added = st.sidebar.selectbox("Select Date ADDED", query_columns['DATE ADDED'].unique())
selected_sn = st.sidebar.selectbox("Select Serial Number", query_columns['SN'].unique())
selected_region = st.sidebar.selectbox("Select REGION", query_columns['REGION'].unique())

# Create a dataframe with user input
user_input_df = pd.DataFrame([user_input])

# Display user input
st.write("User Input:")
st.write(f"Selected Date Added: {selected_date_added}")
st.write(f"Selected Serial Number: {selected_sn}")
st.write(f"Selected REGION: {selected_region}")

# Predict the category using the selected model
selected_model_index = st.sidebar.selectbox("Select a model", range(len(models)))
selected_model = models[selected_model_index]

prediction = selected_model.predict(user_input_df)
st.write(f"Predicted Category: {prediction[0]}")

# Display model evaluation results
st.header("Model Evaluation Results")
for model in models:
    accuracy = cross_val_score(model, balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'], cv=5)
    st.write(f"Accuracy of {model.__class__.__name__}: {accuracy.mean()}")

# Determine CATEGORY based on selected columns
selected_row = query_columns[
    (query_columns['DATE ADDED'] == selected_date_added) &
    (query_columns['SN'] == selected_sn) &
    (query_columns['REGION'] == selected_region)
]

if not selected_row.empty:
    predicted_category = selected_row['CATEGORY'].values[0]
    st.write(f"Predicted Category: {predicted_category}")
else:
    st.warning("No matching row found for the selected columns.")

# Display predictions and probabilities
predictions = selected_model.predict(test_data.drop(['CATEGORY'], axis=1))
probabilities = selected_model.predict_proba(test_data.drop(['CATEGORY'], axis=1))
dosifier_predictions = pd.DataFrame(probabilities, columns=selected_model.classes_, index=test_data.index)
dosifier_predictions_final = dosifier_predictions.groupby(level=0).mean()
st.header("Test Data Predictions and Probabilities")
st.write(dosifier_predictions_final)

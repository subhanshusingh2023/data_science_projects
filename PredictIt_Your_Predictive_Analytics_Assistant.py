# Run this by using `streamlit run PredictIt_Your_Predictive_Analytics_Assistant.py`
#REQUIREMENTS -  pandas , numpy , scikit-learn , matplotlib, seaborn, streamlit

# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Define a function to perform regression with the selected model
def perform_regression(model, file, selected_features, selected_target):
    try:
        # Check if the uploaded file is empty
        if file is None:
            return "Error: Empty CSV file. Please upload a CSV file with data."

        # Read the CSV file from the file object
        df = pd.read_csv(file)

        # Check if the DataFrame is empty
        if df.empty:
            return "Error: The CSV file is empty. Please upload a CSV file with data."

        # Check if the selected features and target column are valid
        if selected_target not in df.columns:
            return "Error: Selected target column not found in the CSV file."
        if not all(feature in df.columns for feature in selected_features):
            return "Error: One or more selected features not found in the CSV file."

        # Select features and target column
        X = df[selected_features]
        y = df[selected_target]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create a dictionary of models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Machine": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(),
        }

        # Perform Grid Search CV to find the best hyperparameters for the selected model
        grid = {
            "Linear Regression": {},
            "Decision Tree": {"max_depth": [None, 10, 20, 30]},
            "Random Forest": {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 10, 20, 30]},
            "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
            "Gradient Boosting": {"n_estimators": [10, 50, 100, 200], "learning_rate": [0.001, 0.01, 0.1]},
        }

        best_params = {}
        if model in models:
            model_instance = models[model]
            grid_search = GridSearchCV(model_instance, grid[model], scoring="neg_mean_squared_error", cv=5)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            model_instance.set_params(**best_params)
            model_instance.fit(X_train, y_train)

            # Make predictions
            y_pred = model_instance.predict(X_test)
            joblib.dump(model_instance, 'trained_model_regression.joblib')

            # Calculate Mean Squared Error (MSE)
            mse = mean_squared_error(y_test, y_pred)

            # Calculate R-squared (R2) score
            r2 = r2_score(y_test, y_pred)

            # Create an Actual vs. Predicted Values Chart
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.5, c='b', label='Predicted Values')
            ax.scatter(y_test, y_test, alpha=0.5, c='r', label='Actual Values')
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title(f"Actual vs. Predicted Values Chart ({model} Model)")
            ax.legend()
            st.pyplot(fig)  # Display the chart in Streamlit

            return f"Model: {model}\nBest Hyperparameters: {best_params}\nMean Squared Error (MSE): {mse:.2f}\nR-squared (R2) Score: {r2:.2f}"

    except Exception as e:
        return f"Error: {str(e)}"
# Define a function to make predictions on unseen data and output the CSV file
def make_predictions_on_unseen_data(model, file, selected_features):
    try:
        # Check if the uploaded file is empty
        if file is None:
            return "Error: Empty CSV file. Please upload a CSV file with data."

        # Read the CSV file from the file object
        data_unseen = pd.read_csv(file)
       

        # Check if the DataFrame is empty
        if data_unseen.empty:
            return "Error: The CSV file is empty. Please upload a CSV file with data."

        # Check if the selected features are valid
        if not all(feature in data_unseen.columns for feature in selected_features):
            return "Error: One or more selected features not found in the CSV file."

        # Standardize features
        scaler = StandardScaler()
        X_unseen = data_unseen[selected_features]
        X_unseen = scaler.fit_transform(X_unseen)     
        y_pred = model.predict(X_unseen)

        # Add predictions as a new column in the DataFrame
        data_unseen['Predicted_Labels'] = y_pred
        output_filename = "predictions_on_unseen_data.csv"
        out_file = data_unseen.to_csv(data_unseen.to_csv(output_filename, index=False))
        return output_filename,out_file
    except Exception as e:
       return f"Error: {str(e)}"

# Create a Streamlit app
st.title("PredictIt - Your Predictive Analytics Assistant")

st.markdown("""
Welcome to PredictIt, your all-in-one predictive analytics assistant! PredictIt allows you to effortlessly make predictions and gain insights from your data using various machine learning models.
""")

st.markdown("""
Whether you're a data enthusiast, business analyst, or just curious about what the future holds, PredictIt empowers you to make data-driven decisions and explore the fascinating world of predictive analytics.
""")
# Data Preparation Guidance
st.markdown("""
### Data Preparation

For accurate predictions, please ensure that your data is cleaned and preprocessed properly. Follow these guidelines:

- Remove any missing or duplicate values.
- Convert categorical columns to numerical using techniques like one-hot encoding or label encoding.
- Upload the clean and preprocessed data for making predictions.

Let's get started with your data analysis and predictions!
""")


# Upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)

    # Display the DataFrame
    st.write("Uploaded Data:")
    st.write(df)

    # User selects features and target column
    selected_features = st.multiselect("Select Features", df.columns.tolist())
    target_column = st.selectbox("Select Target Column", df.columns.tolist())

    # User selects the regression model
    selected_model = st.selectbox("Select Regression Model", ["Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "Gradient Boosting"])

    # Perform regression with the selected model
    if st.button("Predict"):
        metrics = perform_regression(selected_model, uploaded_file, selected_features, target_column)
        st.write(metrics)


# Section for making predictions on unseen data
st.header("Make Predictions on Unseen Data")
st.markdown("""
### Same Format
Please make sure the format of unseen data is same and has the same features
""")

# Upload a CSV file for unseen data
unseen_data_file = st.file_uploader("Upload Unseen Data (without labels)", type=["csv"])

if unseen_data_file is not None:
    st.write("Uploaded Unseen Data:")
    df_unseen = pd.read_csv(unseen_data_file)
    unseen_data_file.seek(0)
    st.write(df_unseen)

    # Button to make predictions on unseen data
    if st.button("Make Predictions on Unseen Data"):
        output_filename,out_file = make_predictions_on_unseen_data(joblib.load('trained_model_regression.joblib'), unseen_data_file, selected_features)
       


        if output_filename:
            st.success(f"Predictions saved to '{output_filename}'")

            # Provide a download link for the user to download the file
            st.download_button(
                label="Download Predictions",
                data=out_file,
                file_name=output_filename,
                key=None,
                help="Click to download the predictions CSV file",
            )
    
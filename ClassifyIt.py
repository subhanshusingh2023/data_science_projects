# Run this by using `streamlit run ClassifyIt.py`
#REQUIREMENTS -  pandas , numpy , scikit-learn , matplotlib, seaborn, streamlit
# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Defined global variables
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(),
}

grid = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 9]},
    "Naive Bayes": {},
    "Decision Tree": {"max_depth": [None, 10, 20, 30]},
    "Random Forest": {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 10, 20, 30]},
    "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "Gradient Boosting": {"n_estimators": [10, 50, 100, 200], "learning_rate": [0.001, 0.01, 0.1]},
}




# Defined a function to perform classification with the selected model
def perform_classification(file, selected_features, selected_target,selected_model):
    

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

        if selected_model in models:  # Check if selected_model is valid
            model_instance = models[selected_model]
            grid_search = GridSearchCV(model_instance, grid[selected_model], cv=5)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            model_instance.set_params(**best_params)
            model_instance.fit(X_train, y_train)

            
            # Make predictions
            y_pred = model_instance.predict(X_test)
            joblib.dump(model_instance, 'trained_model_classification.joblib')
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Generate classification report
            report = classification_report(y_test, y_pred,output_dict=True)

             # Create a confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            parameters =  [selected_model, best_params, accuracy, report, cm]

            return  parameters
    except Exception as e:
        return f"Error: {str(e)}"

# Defined a function to make predictions on unseen data and output the CSV file
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
st.title("ClassifyIt - Your Classification Assistant")

st.title("ClassifyIt - Your Classification Assistant")

st.markdown("""
Welcome to ClassifyIt, your all-in-one classification assistant! ClassifyIt allows you to perform classification tasks and analyze the results using various machine learning models.
""")

st.markdown("""
Whether you're working on binary classification, multi-class classification, or just want to predict categories, ClassifyIt empowers you to make data-driven decisions and explore the world of classification.
""")

# Data Preparation Guidance
st.markdown("""
### Data Preparation

For accurate classification, please ensure that your data is cleaned and preprocessed properly. Follow these guidelines:

- Remove any missing or duplicate values.
- Convert categorical columns to numerical using techniques like one-hot encoding or label encoding.
- Ensure that the target column contains the correct class labels (e.g., 0, 1 for binary classification, or class names for multi-class classification).
- Upload the clean and preprocessed data for classification.

Let's get started with your classification task!
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

    # User selects the classification model
    selected_model = st.selectbox("Select Classification Model", list(models.keys()))

    # Perform classification with the selected model
    if st.button("Classify"):
        metrics = perform_classification(uploaded_file, selected_features, target_column,selected_model)

        # Display classification metrics
        if metrics is not None:
            st.markdown(f"**Model:** {metrics[0]}")
            st.markdown(f"**Best Hyperparameters:** {metrics[1]}")
            st.markdown(f"**Accuracy:** {metrics[2]:.2f}")
            classification_report_df = pd.DataFrame(metrics[3]).T
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(classification_report_df.iloc[:, :3], annot=True, cmap="YlGnBu", cbar=False, fmt=".2f", linewidths=0.5)
            st.pyplot(fig)
            
            if metrics[4] is not None:
            # Create a heatmap of the confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(metrics[4], annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16})
                plt.xlabel('Predicted Labels', fontsize=14)
                plt.ylabel('True Labels', fontsize=14)
                plt.title('Confusion Matrix', fontsize=16)
                st.pyplot(fig)

                    


       

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
        output_filename,out_file = make_predictions_on_unseen_data(joblib.load('trained_model_classification.joblib'), unseen_data_file, selected_features)
       


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

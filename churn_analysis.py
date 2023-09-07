import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

loaded_model = joblib.load("customer_churn_nn.joblib")

grid = {
        "Random Forest": {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 10, 20, 30]},
        }

def perform_classification(X_train,X_test,y_train,y_test):

    model_instance = loaded_model
    grid_search = GridSearchCV(model_instance, grid["Random Forest"], cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    model_instance.set_params(**best_params)
    model_instance.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model_instance.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred,output_dict=True)

     # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    parameters =  [report, cm]

    return  parameters

        

def feature_selection(X,y,k="all"):
     # Select features and target column
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        
        # Initialize StandardScaler
        scaler = StandardScaler()

        # Fit and transform the scaler on training data
        X_train_scaled = scaler.fit_transform(X_train)

        # Transform the test data using the same scaler
        X_test_scaled = scaler.transform(X_test)

        # Initialize SelectKBest with the scoring function (f_classif for classification)
        selector = SelectKBest(score_func=f_classif, k=k)

        # Fit the selector to your training data and transform the features
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

        return X_train_selected, X_test_selected, y_train, y_test
    
# Create a Streamlit title
st.title("Customer Churn Classification [Random Forest]")

# Add instructions for the user
st.write("Upload a CSV file containing your data. Select features and the target column for classification, then click 'Classify'.")

# Add a markdown message for data preprocessing
st.markdown("Before uploading your data, make sure it's preprocessed and in the right format.")
    
uploaded_file = st.file_uploader("Upload a CSV File",type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)

    # User selects features and target column
    selected_features = st.multiselect("Select Features", data.columns.tolist())
    target_column = st.selectbox("Select Target Column", data.columns.tolist())
    X = data[selected_features]
    y = data[target_column]

    if st.button("Classify"):
        X_train,X_test,y_train,y_test = feature_selection(X,y)

        performance = perform_classification(X_train,X_test,y_train,y_test)

        if performance is not None:
            classification_report_df = pd.DataFrame(performance[0]).T
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(classification_report_df.iloc[:, :3], annot=True, cmap="YlGnBu", cbar=False, fmt=".2f", linewidths=0.5)

            st.pyplot(fig)

        if performance is not None:
        # Create a heatmap of the confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(performance[1], annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16})
            plt.xlabel('Predicted Labels', fontsize=14)
            plt.ylabel('True Labels', fontsize=14)
            plt.title('Confusion Matrix', fontsize=16)
            st.pyplot(fig)





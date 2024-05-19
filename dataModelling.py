import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Perceptron, HuberRegressor, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import plotly.express as px
import os
from utils.qbutils import fetch_qbo_data
from utils.db_utils import create_connection, fetch_user_datasets, insert_dataset

def convert_date_to_numeric(df):
    date_columns = df.select_dtypes(include=['object']).columns
    for date_column in date_columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column], format='%b-%y')
        except ValueError:
            try:
                df[date_column] = pd.to_datetime(df[date_column])
            except ValueError:
                st.write(f"Column {date_column} could not be converted to datetime. Skipping conversion.")
                continue
        df[date_column] = df[date_column].apply(lambda x: x.timestamp())
    return df

def show_data_analysis_page():
    user_id = st.session_state.get("user_id")  # Assume user_id is stored in session state
    
    # Create or connect to the database
    conn = create_connection("user_data.db")

    # Create tables if they don't exist
    if conn is None:
        st.error("Error: Could not connect to the database.")

    # Title and description
    st.title("Data Modeling and Prediction Tool")
    st.write("This tool allows you to upload a CSV file, visualize data, choose variables and transformations, select a model, and make predictions.")

    # Function to check if data pattern is classification or regression
    def check_data_pattern(y_values):
        unique_values = np.unique(y_values)
        if len(unique_values) <= 10 and y_values.dtype in ['int64', 'object']:
            return "classification"
        else:
            return "regression"

    selected_dataset = None 
    df = None

    # Data input section
    st.header("Data Input")
    # Option to choose existing dataset or upload new dataset
    data_option = st.radio("Choose data option:", options=["Use existing dataset", "Upload new dataset", "Fetch data from QuickBooks API"])

    if data_option == "Use existing dataset":
        # Fetch and display user's datasets
        st.subheader("Select a saved dataset:")
        datasets = fetch_user_datasets(conn, user_id)

        if datasets and len(datasets) > 0:
            dataset_names = [dataset["dataset_name"] for dataset in datasets]
            selected_dataset_name = st.selectbox("Select a dataset", options=dataset_names)
            selected_dataset = next((ds for ds in datasets if ds["dataset_name"] == selected_dataset_name), None)
            
            if selected_dataset:
                try:
                    df = pd.read_csv(selected_dataset["dataset_path"])
                    st.write(f"Dataset: {selected_dataset_name}")
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")
        else:
            st.write("You have no datasets saved.")

    elif data_option == "Upload new dataset":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            dataset_name = st.text_input("Dataset Name")
            if st.button("Save Dataset"):
                # Create the datasets directory if it doesn't exist
                os.makedirs("datasets", exist_ok=True)
                dataset_path = f"datasets/{dataset_name}.csv"  # Define a path to save the dataset
                df.to_csv(dataset_path, index=False)
                dataset_id = insert_dataset(conn, user_id, dataset_name, dataset_path)
                if dataset_id:
                    st.success(f"Dataset '{dataset_name}' saved successfully!")
                else:
                    st.error("Failed to save dataset.")

    elif data_option == "Fetch data from QuickBooks API":
        # Get user credentials or access token for QuickBooks API
        realm_id = st.text_input("Realm ID")
        access_token = st.text_input("Access Token")

        # Choose the report and date range
        report_name = st.selectbox("Select report:", options=["ProfitAndLoss", "BalanceSheet"])
        start_date = st.date_input("Start date")
        end_date = st.date_input("End date")

        if st.button("Fetch Data"):
            if not realm_id or not access_token or not report_name or not start_date or not end_date:
                st.error("Please fill in all the required fields.")
            elif start_date > end_date:
                st.error("Start date cannot be after end date.")
            else:
                # Fetch data from QuickBooks API
                try:
                    df = fetch_qbo_data(realm_id, access_token, report_name, start_date, end_date)
                    if df.empty:
                        st.error("The fetched dataset is empty. Please check your credentials and try again.")
                    else:
                        st.success("Data fetched successfully!")
                        st.write(df)
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

    if selected_dataset or df is not None and not df.empty:
        st.write("Original Data:")
        st.write(df)

        # Convert date columns to datetime
        df = convert_date_to_numeric(df)

        # Convert date columns to datetime
        date_columns = df.select_dtypes(include=['object']).columns
        for date_column in date_columns:
            try:
                df[date_column] = pd.to_datetime(df[date_column])
            except ValueError:
                st.write(f"Column {date_column} could not be converted to datetime. Skipping conversion.")
        
        # Convert datetime columns to numerical format (e.g., timestamp)
        for date_column in date_columns:
            if pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = df[date_column].apply(lambda x: x.timestamp())

        # Data plot section
        st.header("Data Plot")
        st.subheader("Select plot type:")
        plot_type = st.selectbox("Plot type", options=["Scatter", "Line", "Bar"])

        st.subheader("Select variables for plot:")
        x_variable = st.selectbox("X-axis variable", options=df.columns)
        y_variable = st.selectbox("Y-axis variable", options=df.columns)

        if st.button("Plot"):
            if plot_type == "Scatter":
                fig = px.scatter(df, x=x_variable, y=y_variable, title="Scatter Plot")
            elif plot_type == "Line":
                fig = px.line(df, x=x_variable, y=y_variable, title="Line Plot")
            elif plot_type == "Bar":
                fig = px.bar(df, x=x_variable, y=y_variable, title="Bar Plot")
            st.plotly_chart(fig)

        # Data modeling section
        st.header("Data Modeling")
        train_test_ratio = st.slider("Training data ratio (%)", min_value=0, max_value=100, value=80)

        # Display number of rows for training and testing sets dynamically
        train_size = int(len(df) * train_test_ratio / 100)
        test_size = len(df) - train_size
        st.write(f"Training set: {train_size} rows")
        st.write(f"Testing set: {test_size} rows")

        shuffle_data = st.checkbox("Shuffle data before splitting", value=True)

        # Determine if the dataset is for classification or regression
        st.subheader("Select Dependent Variable:")
        y_variable_choice = st.selectbox("Choose the dependent variable", options=df.columns)
        task_type = check_data_pattern(df[y_variable_choice])

        # Filter out non-numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        x_variables = [col for col in numeric_columns if col != x_variable and col != y_variable]

        # Remove dependent variable from transformation options
        x_variables_transformations = {col: ["identity", "sine", "cosine", "square", "root", "ignore"] for col in x_variables}
        if y_variable_choice in x_variables_transformations:
            del x_variables_transformations[y_variable_choice]

        st.subheader("Variable transformations:")
        transformations = {}
        for variable, transformation_options in x_variables_transformations.items():
            transformation = st.selectbox(f"Transformation for {variable}", options=transformation_options)
            transformations[variable] = transformation

        # Model selection based on task type
        if task_type == "classification":
            models = ["Perceptron", "Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier"]
        else:
            models = ["Linear Regression", "HuberRegressor", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor"]

        # Model selection
        st.subheader("Select Model:")
        model_name = st.selectbox("Choose a model", options=models)

        if st.button("Submit"):
            # Drop rows with NaN in y values
            df = df.dropna(subset=[y_variable_choice])

            # Split data
            X = df[x_variables]
            y = df[y_variable_choice]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_test_ratio) / 100, random_state=42, shuffle=shuffle_data)

            # Display number of rows for training and testing sets
            st.write(f"Training set: {X_train.shape[0]} rows")
            st.write(f"Testing set: {X_test.shape[0]} rows")

            # Impute missing values
            imputer = SimpleImputer(strategy='mean')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            # Apply transformations
            for variable, transformation in transformations.items():
                col_idx = df.columns.get_loc(variable)
                if col_idx < X_train.shape[1]: 
                    if transformation == "sine":
                        X_train[:, df.columns.get_loc(variable)] = np.sin(X_train[:, df.columns.get_loc(variable)])
                        X_test[:, df.columns.get_loc(variable)] = np.sin(X_test[:, df.columns.get_loc(variable)])
                    elif transformation == "cosine":
                        X_train[:, df.columns.get_loc(variable)] = np.cos(X_train[:, df.columns.get_loc(variable)])
                        X_test[:, df.columns.get_loc(variable)] = np.cos(X_test[:, df.columns.get_loc(variable)])
                    elif transformation == "square":
                        X_train[:, df.columns.get_loc(variable)] = X_train[:, df.columns.get_loc(variable)] ** 2
                        X_test[:, df.columns.get_loc(variable)] = X_test[:, df.columns.get_loc(variable)] ** 2
                    elif transformation == "root":
                        X_train[:, df.columns.get_loc(variable)] = np.sqrt(X_train[:, df.columns.get_loc(variable)])
                        X_test[:, df.columns.get_loc(variable)] = np.sqrt(X_test[:, df.columns.get_loc(variable)])
                    elif transformation == "ignore":
                        X_train = np.delete(X_train, df.columns.get_loc(variable), axis=1)
                        X_test = np.delete(X_test, df.columns.get_loc(variable), axis=1)

            # Fit model
            if task_type == "classification":
                if model_name == "Perceptron":
                    model = Perceptron()
                elif model_name == "Logistic Regression":
                    model = LogisticRegression()
                elif model_name == "Decision Tree Classifier":
                    model = DecisionTreeClassifier()
                elif model_name == "Random Forest Classifier":
                    model = RandomForestClassifier()
                elif model_name == "Support Vector Classifier":
                    model = SVC()           
            else:
                if model_name == "Support Vector Classifier":
                    model = SVC()
                elif model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "HuberRegressor":
                    model = HuberRegressor()
                elif model_name == "Decision Tree Regressor":
                    model = DecisionTreeRegressor()
                elif model_name == "Random Forest Regressor":
                    model = RandomForestRegressor()
                elif model_name == "Support Vector Regressor":
                    model = SVR()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if task_type == "classification":
                st.metric("Accuracy", accuracy_score(y_test, y_pred))
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))
            else:
                st.metric("Mean Squared Error", mean_squared_error(y_test, y_pred))
                st.metric("R^2", r2_score(y_test, y_pred))

            # Display results
            st.subheader("Model Performance:")
            st.write("Predictions vs Actual:")
            results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            st.dataframe(results, width=700)

            # Plot predicted vs actual using Plotly
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Predicted vs Actual")
            fig.add_shape(
                type="line",
                x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
                line=dict(color="red", dash="dash")
            )
            st.plotly_chart(fig)
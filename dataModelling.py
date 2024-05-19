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
from utils.db_utils import create_connection, fetch_user_datasets

def show_data_analysis_page():
   
    user_id = st.session_state.get("user_id")  # Assume user_id is stored in session state
    
    # Create or connect to the database
    conn = create_connection("user_data.db")

    # Create tables if they don't exist
    if conn is None:
        st.error("Error: Could not connect to the database.")

    def check_data_pattern(y_values):
        unique_values = np.unique(y_values)
        if len(unique_values) <= 10 and y_values.dtype in ['int64', 'object']:
            return "classification"
        else:
            return "regression"

    def convert_dates(df):
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and df[col].str.contains('-').any():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col] = (df[col] - pd.Timestamp('1970-01-01')).dt.days
                except Exception as e:
                    st.write(f"Error converting column {col} to date: {e}")
        return df

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
            df = convert_dates(df)

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

        st.subheader("Select Dependent Variable:")
        y_variable_choice = st.selectbox("Choose the dependent variable", options=df.columns)

        # Remove dependent variable from transformation options
        x_variables = [col for col in df.columns if col != y_variable_choice]
        x_variables_transformations = {col: ["identity", "sine", "cosine", "square", "root", "ignore"] for col in x_variables}

        st.subheader("Variable transformations:")
        transformations = {}
        for variable, transformation_options in x_variables_transformations.items():
            transformation = st.selectbox(f"Transformation for {variable}", options=transformation_options)
            transformations[variable] = transformation

        # Determine if the task is classification or regression
        task_type = check_data_pattern(df[y_variable_choice])

        st.subheader("Select Model:")
        if task_type == "classification":
            model_name = st.selectbox("Choose a model", options=["Perceptron", "Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier"])
        else:
            model_name = st.selectbox("Choose a model", options=["Linear Regression", "HuberRegressor", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor"])

    if st.button("Submit"):
        # Drop rows with NaN in y values
        df = df.dropna(subset=[y_variable_choice])

        if len(df) > 0:  # Check if the dataset is not empty
            # Split data
            X = df[x_variables]
            y = df[y_variable_choice]
            
            if len(X) > 0 and len(y) > 0:  # Check if both X and y contain samples
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_test_ratio) / 100, random_state=42, shuffle=shuffle_data)
                
                if len(X_train) > 0 and len(X_test) > 0 and len(y_train) > 0 and len(y_test) > 0:  # Check if train and test sets are not empty
                    # Display number of rows for training and testing sets
                    st.write(f"Training set: {X_train.shape[0]} rows")
                    st.write(f"Testing set: {X_test.shape[0]} rows")

                    # Preprocess the data
                    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                    categorical_features = X.select_dtypes(include=['object']).columns

                    numeric_transformer = SimpleImputer(strategy='mean')
                    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numeric_transformer, numeric_features),
                            ('cat', categorical_transformer, categorical_features)
                        ])

                    X_train = preprocessor.fit_transform(X_train)
                    X_test = preprocessor.transform(X_test)

                    # Apply transformations
                    for variable, transformation in transformations.items():
                        col_idx = df.columns.get_loc(variable)
                        if col_idx < X_train.shape[1]:  # Check if the column index is within the bounds
                            if transformation == "sine":
                                X_train[:, col_idx] = np.sin(X_train[:, col_idx])
                                X_test[:, col_idx] = np.sin(X_test[:, col_idx])
                            elif transformation == "cosine":
                                X_train[:, col_idx] = np.cos(X_train[:, col_idx])
                                X_test[:, col_idx] = np.cos(X_test[:, col_idx])
                            elif transformation == "square":
                                X_train[:, col_idx] = X_train[:, col_idx] ** 2
                                X_test[:, col_idx] = X_test[:, col_idx] ** 2
                            elif transformation == "root":
                                X_train[:, col_idx] = np.sqrt(X_train[:, col_idx])
                                X_test[:, col_idx] = np.sqrt(X_test[:, col_idx])
                            elif transformation == "ignore":
                                X_train = np.delete(X_train, col_idx, axis=1)
                                X_test = np.delete(X_test, col_idx, axis=1)

                    # Fit model and make predictions
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

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.metric("Accuracy", accuracy_score(y_test, y_pred))
                        st.write("Classification Report:")
                        st.text(classification_report(y_test, y_pred))
                    else:
                        if model_name == "Linear Regression":
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
                else:
                    st.warning("Train or test set is empty after splitting. Adjust the parameters.")
            else:
                st.warning("X or y contains no samples. Adjust the parameters.")
        else:
            st.warning("Datframe is empty. Adjust the parameters.")

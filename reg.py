import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import plotly.express as px
import os
from utils.qbutils import fetch_qbo_data
from utils.db_utils import create_connection, fetch_user_datasets, insert_dataset
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def convert_date_to_numeric(df):
    """Converts date columns to numeric (timestamp) format."""
    date_columns = df.select_dtypes(include=['object']).columns
    for date_column in date_columns:
        try:
            # Attempt to parse dates with format '%b-%y' (e.g., 'Jan-19')
            df[date_column] = pd.to_datetime(df[date_column], format='%b-%y', errors='coerce')
            # If any dates are still 'NaT', try parsing without specifying the format
            if df[date_column].isnull().any():
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            # Convert datetime to timestamp, handling 'NaT' values
            df[date_column] = df[date_column].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
        except Exception as e:
            st.warning(f"Could not convert {date_column} to datetime: {e}")
    return df

def calculate_future_predictions(last_values, selected_transformations, increase_percentages, future_periods):
    """Calculates future predictions based on percentage increases."""
    future_data_list = []

    for period in range(1, future_periods + 1):
        future_point = {}
        for var in last_values.keys():
            increase_factor = (1 + increase_percentages[var] / 100) ** period
            base_value = last_values[var] * increase_factor
            future_point[var] = base_value
        future_data_list.append(future_point)
    
    future_data_df = pd.DataFrame(future_data_list)
    return future_data_df

def reset_session_state_variables():
    """Resets session state variables related to model and data."""
    session_vars = ['model_trained', 'model', 'transformations', 'x_variables', 'preprocessor']
    for var in session_vars:
        if var in st.session_state:
            del st.session_state[var]

def show_data_analysis_page():
    """Displays the data analysis page with various functionalities."""
    user_id = st.session_state.get("user_id")
    
    # Create or connect to the database
    conn = create_connection("user_data.db")

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

    # Initialize or reset session state variables
    if 'last_selected_dataset' not in st.session_state:
        st.session_state['last_selected_dataset'] = None

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

            # Reset session state if dataset changes
            if st.session_state['last_selected_dataset'] != selected_dataset_name:
                reset_session_state_variables()
                st.session_state['last_selected_dataset'] = selected_dataset_name

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

    if df is not None and not df.empty:
        st.write("Original Data:")
        st.write(df)


        df = convert_date_to_numeric(df)

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

        # Select independent variables
        st.subheader("Select Independent Variables:")
        x_variables = st.multiselect("Select independent variables", options=[col for col in df.columns if col != y_variable_choice])

        # Variable transformations
        st.subheader("Variable transformations:")
        transformations = {}
        for variable in x_variables:
            transformation = st.selectbox(f"Transformation for {variable}", options=["identity", "sine", "cosine", "square", "root", "ignore"], key=variable)
            transformations[variable] = transformation

        # Model selection based on task type
        if task_type == "classification":
            models = ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier"]
        else:
            models = ["Linear Regression", "HuberRegressor", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor"]

        # Model selection
        st.subheader("Select Model:")
        model_name = st.selectbox("Choose a model", options=models)
        
        with st.form(key="model_form"):
            submit_button = st.form_submit_button("Train Model")
            if submit_button:
                # Reset session state variables before training new model
                reset_session_state_variables()

                # Drop rows with NaN in y values
                df = df.dropna(subset=[y_variable_choice])

                # Convert y to numeric by removing commas and quotes
                df[y_variable_choice] = pd.to_numeric(df[y_variable_choice].astype(str).str.replace(',', '').str.replace('"', ''), errors='coerce')

                # Drop any rows where conversion failed
                df = df.dropna(subset=[y_variable_choice])

                # Split data
                X = df[x_variables]
                y = df[y_variable_choice]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=(100 - train_test_ratio) / 100,
                    random_state=42,
                    shuffle=shuffle_data
                )

                # Display number of rows for training and testing sets
                st.write(f"Training set: {X_train.shape[0]} rows")
                st.write(f"Testing set: {X_test.shape[0]} rows")

                # Separate numeric and categorical columns
                numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

                # Update transformations to exclude ignored variables
                numeric_cols = [col for col in numeric_cols if transformations[col] != "ignore"]
                categorical_cols = [col for col in categorical_cols if transformations[col] != "ignore"]

                # Define transformers for numeric and categorical data
                def custom_numeric_transform(X):
                    X_transformed = X.copy()
                    for idx, variable in enumerate(numeric_cols):
                        transformation = transformations.get(variable, "identity")
                        if transformation == "sine":
                            X_transformed[:, idx] = np.sin(X_transformed[:, idx])
                        elif transformation == "cosine":
                            X_transformed[:, idx] = np.cos(X_transformed[:, idx])
                        elif transformation == "square":
                            X_transformed[:, idx] = X_transformed[:, idx] ** 2
                        elif transformation == "root":
                            # To handle negative values for square root, replace negative numbers with zero or another appropriate value
                            X_transformed[:, idx] = np.where(X_transformed[:, idx] >= 0, np.sqrt(X_transformed[:, idx]), 0)
                        # 'identity' does nothing
                    return X_transformed

                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('transformer', FunctionTransformer(custom_numeric_transform))
                ])

                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])

                # Combine transformations using ColumnTransformer
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_cols),
                        ('cat', categorical_transformer, categorical_cols)
                    ]
                )

                # Fit and transform training data, transform testing data
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)

                # Fit model
                if task_type == "classification":
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    elif model_name == "Decision Tree Classifier":
                        model = DecisionTreeClassifier()
                    elif model_name == "Random Forest Classifier":
                        model = RandomForestClassifier()
                    elif model_name == "Support Vector Classifier":
                        model = SVC()
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

                model.fit(X_train_processed, y_train)
                y_pred = model.predict(X_test_processed)

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

                # Save model and other variables in session state
                st.session_state['model_trained'] = True
                st.session_state['model'] = model
                st.session_state['preprocessor'] = preprocessor
                st.session_state['transformations'] = transformations
                st.session_state['x_variables'] = x_variables
                st.session_state['numeric_cols'] = numeric_cols
                st.session_state['categorical_cols'] = categorical_cols

    # Future Predictions section (outside the form)
    if st.session_state.get('model_trained', False):
        st.header("Future Predictions")
        future_periods = st.number_input("Enter the number of future periods to predict:", min_value=1, value=10)

        st.write("Enter the expected percentage increase or decrease for each variable (per period):")
        increase_percentages = {}
        for var in st.session_state['x_variables']:
            if st.session_state['transformations'][var] != "ignore":
                increase_percentages[var] = st.number_input(f"Percentage change for {var} per period (%):", value=0.0)

        st.subheader("Initial Values for Variables")
        last_values = {}
        for var in st.session_state['x_variables']:
            if st.session_state['transformations'][var] != "ignore":
                last_values[var] = st.number_input(f"Initial value for {var}:", value=0.0)

        if st.button("Calculate Future Predictions"):
            # Calculate future predictions based on selected increments
            future_data_df = calculate_future_predictions(
                last_values,
                st.session_state['transformations'],
                increase_percentages,
                int(future_periods)
            )

            # Apply the same preprocessing to future data
            X_future = st.session_state['preprocessor'].transform(future_data_df)

            # Make predictions
            future_predictions = st.session_state['model'].predict(X_future)

            # Constrain predictions to be non-negative
            future_predictions = np.maximum(future_predictions, 0)

            # Assign predictions to the DataFrame
            future_data_df['Predicted Value'] = future_predictions

            # Ensure all columns are numeric
            for col in future_data_df.columns:
                future_data_df[col] = pd.to_numeric(future_data_df[col], errors='coerce')

            # Display future predictions with formatted numbers
            st.subheader("Future Predicted Values:")
            st.write(future_data_df.style.format('{:,.2f}'))

            # Visualization
            num_vars = len(st.session_state['x_variables']) + 1  # Plus one for Predicted Value

            # Create subplots
            fig = make_subplots(
                rows=num_vars,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=st.session_state['x_variables'] + ['Predicted Value']
            )

            # Add traces for each independent variable
            for i, var in enumerate(st.session_state['x_variables']):
                fig.add_trace(
                    go.Scatter(
                        x=future_data_df.index + 1,
                        y=future_data_df[var],
                        mode='lines+markers',
                        name=var
                    ),
                    row=i+1,
                    col=1
                )

            # Add trace for Predicted Value
            fig.add_trace(
                go.Scatter(
                    x=future_data_df.index + 1,
                    y=future_data_df['Predicted Value'],
                    mode='lines+markers',
                    name='Predicted Value'
                ),
                row=num_vars,
                col=1
            )

            fig.update_layout(
                height=200*num_vars,
                title='Future Predictions and Variables',
                xaxis_title='Future Periods',
                yaxis_title='Values'
            )
            st.plotly_chart(fig)
    else:
        st.info("Please train the model first to make future predictions.")

# Call the main function to display the page
show_data_analysis_page()

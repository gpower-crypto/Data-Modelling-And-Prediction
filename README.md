# Machine Learning Based Revenue Forecasting Tool

Feinstein AI's Machine Learning Based Revenue Forecasting Tool is designed to assist startups in predicting future cash reserves based on historical data of revenue, expenses, and existing cash holdings. This tool leverages various machine learning models, such as linear regression, to provide insights that enable strategic financial decisions.

## Short Description

Feinstein AI offers financial advisory services tailored for startups. This tool allows users to upload past financial data or fetch it via the Intuit QuickBooks API. Users can select variables, choose from different ML models, and input future revenue and expense estimates to forecast their expected cash reserves.

## Features

- **Frontend**: Developed using Streamlit, providing an intuitive user interface for:
  - Uploading datasets
  - Selecting independent and dependent variables
  - Choosing machine learning models

- **Backend**: Built with Python and scikit-learn for:
  - Data preprocessing and analysis
  - Model training and prediction
  - Integration with sqlite database for user profile and data storage

- **API Integration**: Utilizes Intuit QuickBooks API to fetch revenue data seamlessly.

## Goal

The goal of this tool is to empower startups to forecast their cash reserves accurately and make informed financial decisions accordingly.

## System and Data Flow

1. **Data Input**: Users upload historical revenue data or fetch it via QuickBooks API.
2. **Data Processing**: Python scripts process the input data, including:
   - Cleaning and preprocessing
   - Feature selection
3. **Model Selection**: Users can select independent and dependent variables and choose an ML model (e.g., linear regression) through the Streamlit interface.
4. **Model Training**: The chosen model is trained using the processed data.
5. **Prediction**: Future revenue predictions are generated based on user-provided estimates of future expenses and revenue.
6. **Output**: The tool provides forecasts of expected cash reserves at specific future points in time.

## Setup Instructions

To run the Machine Learning Based Revenue Forecasting Tool locally, follow these steps:

1. **Clone the repository**:
   ```
   git clone https://github.com/your-username/revenue-forecasting-tool.git
   cd revenue-forecasting-tool
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```
   streamlit run main.py
   ```

4. **Access the application**:
   Open a web browser and navigate to `http://localhost:8501`.

## Contributing

Contributions are welcome! If you have suggestions, feature requests, or bug reports, please open an issue or create a pull request.

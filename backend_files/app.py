import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify

# Initialize Flask app
superkart_sales_api = Flask("SuperKart Sales Predictor")

# Load the trained SuperKart sales prediction model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page
@superkart_sales_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint to predict sales for a single product/store instance
@superkart_sales_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    input_json = request.get_json()

    # Extract relevant features from the input data
    sample = {
        'Product_Weight': input_json['Product_Weight'],
        'Product_Sugar_Content': input_json['Product_Sugar_Content'],
        'Product_Allocated_Area': input_json['Product_Allocated_Area'],
        'Product_Type': input_json['Product_Type'],
        'Product_MRP': input_json['Product_MRP'],
        'Store_Size': input_json['Store_Size'],
        'Store_Location_City_Type': input_json['Store_Location_City_Type'],
        'Store_Type': input_json['Store_Type'],
        'Product_Category': input_json['Product_Category'],
        'Store_Age': input_json['Store_Age']
    }

    # Convert the extracted data into a DataFrame
    input_df = pd.DataFrame([sample])

    # Make sales prediction using the trained model
    prediction = model.predict(input_df).tolist()[0]

    # Return the predicted sales total as a JSON response
    return jsonify({'Predicted_Sales_Total': round(prediction, 2)})

# Define an endpoint to predict sales for a batch of records
@superkart_sales_api.post('/v1/predict_batch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_df = pd.read_csv(file)

    # Preserve identifiers if present in batch file
    product_ids = input_df['Product_Id'].tolist() if 'Product_Id' in input_df.columns else [f"Product_{i}" for i in range(len(input_df))]

    # Perform the exact same Feature Engineering as done during training
    # 1. Map Product_Category from the first 2 characters of Product_Id
    if 'Product_Id' in input_df.columns:
        input_df['Product_Category'] = input_df['Product_Id'].str[:2].map({
            'FD': 'Food',
            'NC': 'Non-Consumable',
            'DR': 'Drinks'
        })
    else:
        # Default fallback if Product_Id is missing
        input_df['Product_Category'] = 'Food'

    # 2. Calculate Store_Age from Store_Establishment_Year
    if 'Store_Establishment_Year' in input_df.columns:
        current_year = 2026
        input_df['Store_Age'] = current_year - input_df['Store_Establishment_Year']
    else:
        # Default fallback if year is missing
        input_df['Store_Age'] = 15

    # 3. Standardize 'reg' -> 'Regular' for Sugar Content
    if 'Product_Sugar_Content' in input_df.columns:
        input_df['Product_Sugar_Content'] = input_df['Product_Sugar_Content'].replace({'reg': 'Regular'})

    # Keep only the columns expected by the preprocessor/pipeline model
    expected_features = [
        'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
        'Product_Type', 'Product_MRP', 'Store_Size',
        'Store_Location_City_Type', 'Store_Type', 'Product_Category', 'Store_Age'
    ]

    # Fill missing columns with defaults if any are missing from the uploaded CSV
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = np.nan

    features_df = input_df[expected_features]

    # Make predictions for the batch data
    predictions = model.predict(features_df).tolist()

    # Create dictionary mapping ID to predicted sales revenue
    output_dict = dict(zip(product_ids, [round(p, 2) for p in predictions]))

    return jsonify(output_dict)

# Run the Flask app
if __name__ == '__main__':
    superkart_sales_api.run(host='0.0.0.0', port=5000, debug=True)

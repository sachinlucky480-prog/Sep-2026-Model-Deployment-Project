import joblib
import pandas as pd
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
        'Store_Id': input_json['Store_Id'],
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
    product_ids = input_df['Product_Id'].tolist() if 'Product_Id' in input_df.columns else range(len(input_df))
    
    # Drop raw columns if present to align with feature engineering
    features_df = input_df.drop(columns=['Product_Id', 'Store_Establishment_Year'], errors='ignore')

    # Make predictions for the batch data
    predictions = model.predict(features_df).tolist()

    # Create dictionary mapping ID to predicted sales revenue
    output_dict = dict(zip(product_ids, [round(p, 2) for p in predictions]))

    return jsonify(output_dict)

# Run the Flask app
if __name__ == '__main__':
    superkart_sales_api.run(host='0.0.0.0', port=5000, debug=True)

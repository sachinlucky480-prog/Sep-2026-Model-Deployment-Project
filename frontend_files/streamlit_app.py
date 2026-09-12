import streamlit as st
import requests
import pandas as pd
import json

# Set page title and layout
st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")

st.title("SuperKart Sales Revenue Predictor")
st.markdown("Predict sales revenues of outlets for the upcoming quarter using the optimized Tuned Random Forest model.")

# Define backend API host URL
BACKEND_URL = "http://superkart-backend:5000"

tab1, tab2 = st.tabs(["Single Prediction (Online)", "Batch Prediction"])

with tab1:
    st.header("Single Item & Store Assessment")

    # Layout into 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        product_weight = st.number_input("Product Weight (kg)", min_value=1.0, max_value=30.0, value=12.5, step=0.1)
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_allocated_area = st.number_input("Allocated Display Area", min_value=0.0, max_value=1.0, value=0.05, format="%.4f")
        product_mrp = st.number_input("Product MRP (Maximum Retail Price)", min_value=10.0, max_value=500.0, value=140.0, step=0.5)

    with col2:
        store_id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location = st.selectbox("Store Location (City Type)", ["Tier 1", "Tier 2", "Tier 3"])

    with col3:
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
        product_category = st.selectbox("Product Category", ["Food", "Drinks", "Non-Consumable"])
        product_type = st.selectbox("Product Type", [
            "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods",
            "Dairy", "Baking Goods", "Canned", "Health and Hygiene",
            "Soft Drinks", "Meat", "Breads", "Hard Drinks", "Others",
            "Starchy Foods", "Breakfast", "Seafood"
        ])
        store_age = st.number_input("Store Age (Years)", min_value=1, max_value=100, value=15, step=1)

    # Prediction Request
    if st.button("Predict Sales Total"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": product_allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Id": store_id,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location,
            "Store_Type": store_type,
            "Product_Category": product_category,
            "Store_Age": store_age
        }

        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"### Predicted Sales Total: ${result['Predicted_Sales_Total']:,.2f}")
            else:
                st.error(f"Backend returned error code: {response.status_code}")
        except Exception as e:
            st.error(f"Could not reach the backend container. Error details: {e}")

with tab2:
    st.header("Bulk Inventory Upload & Forecast (Batch)")
    st.markdown("Upload a batch CSV file to get prediction results for all items.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read and display preview
        preview_df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded CSV Data Preview")
        st.dataframe(preview_df.head(10))

        # Reset file pointer for posting
        uploaded_file.seek(0)

        if st.button("Run Batch Predictions"):
            try:
                files = {'file': uploaded_file.getvalue()}
                response = requests.post(f"{BACKEND_URL}/v1/predict_batch", files=files)

                if response.status_code == 200:
                    results = response.json()

                    # Append results back to the dataframe if keys match row indexes or IDs
                    st.success("Batch predictions compiled successfully!")
                    results_df = pd.DataFrame(list(results.items()), columns=['Product_Id', 'Predicted_Sales_Total'])
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)
                else:
                    st.error(f"Backend returned status code: {response.status_code}")
            except Exception as e:
                st.error(f"Error performing batch prediction: {e}")

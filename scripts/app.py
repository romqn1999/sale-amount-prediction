from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from feature_engineering import LagFeatures, FillNA

# Load the model and data
model = joblib.load('models/xgboost_model.pkl')
data_merged = pd.read_csv('data/data_merged.csv')

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    item_id = int(request.args.get('item_id'))

    # Prepare input data
    input_data = data_merged[data_merged['item_id'] == item_id].copy()

    if input_data.empty:
        return jsonify({'error': 'Invalid item_id'})

    # Add new month for prediction
    unique_shops = input_data[['shop_id', 'item_category_id']].drop_duplicates().sort_values(by='shop_id')
    
    new_month_data = []
    for _, row in unique_shops.iterrows():
        new_month_data.append({
            'date_block_num': 34,
            'shop_id': row['shop_id'],
            'item_id': item_id,
            'month': 11,
            'item_category_id': row['item_category_id']
        })
    new_month_df = pd.DataFrame(new_month_data)

    # Combine with historical data
    combined_data = pd.concat([input_data, new_month_df], ignore_index=True)

    # Generate features
    combined_data = LagFeatures(lags=[1, 2, 3], col='item_cnt_month').transform(combined_data)
    combined_data = LagFeatures(lags=[1], col='avg_item_price').transform(combined_data)
    combined_data = FillNA().transform(combined_data)

    # Prepare the features for prediction
    features = combined_data[combined_data['date_block_num'] == 34].drop(columns=['item_cnt_month', 'avg_item_price', 'date_block_num'])
    
    # Predict the sales for November 2015
    prediction = model.predict(features)

    # Round predictions to nearest integer
    prediction = np.round(prediction).astype(int)

    # Prepare the JSON response
    response = []
    for shop_id, predicted_sales in zip(features['shop_id'], prediction):
        response.append({'shop_id': int(shop_id), 'predicted_sales': int(predicted_sales)})

    return jsonify({'item_id': item_id, 'predictions': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

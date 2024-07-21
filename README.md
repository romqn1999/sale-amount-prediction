# Monthly Sales Forecasting

[This repository](https://github.com/romqn1999/sale-amount-prediction) contains a solution for forecasting the monthly sales of products in different shops using historical sales data. The project is divided into two main tasks: model training and deployment.

## Task 1: Model Training

### Summary

You are provided with historical sales data on a daily basis. You need to forecast the total amount of products sold in every shop for the test set, which refers to November 2015. The notebook includes data processing, feature engineering, model training, and evaluation.

### Notebook

The [`notebooks/Experiment.ipynb`](./notebooks/Experiment.ipynb) notebook contains the following steps:

1. **Data Loading and Preprocessing**: Load and preprocess the historical sales data.
2. **Feature Engineering**: Create lag features to capture the trends and patterns in the data.
3. **Model Training**: Train an XGBoost model using GridSearchCV to find the best hyperparameters.
4. **Evaluation**: Evaluate the model using RMSE (Root Mean Squared Error) on a validation set.
5. **Saving the Model**: Save the trained model and processed data for deployment.

Open [`Experiment.ipynb`](./notebooks/Experiment.ipynb) and run all cells to generate the trained model and save the necessary files in the `models/` and `data/` directories.

## Task 2: Model Deployment

### Deployment Patterns

For deploying machine learning models, there are several common patterns, including:

1. **Batch Prediction**: This pattern involves running the model at scheduled intervals to generate predictions for a batch of data. It's useful when real-time predictions are not required, and predictions can be generated in advance.

2. **Real-time Prediction**: This pattern involves serving the model through an API, allowing users to request predictions on-demand. It's suitable for applications requiring immediate predictions based on user input.

3. **Embedded Model**: This pattern involves embedding the model directly into an application, such as a mobile app or desktop software, allowing it to make predictions locally without a server.

4. **Streaming Prediction**: This pattern involves integrating the model with a streaming platform to make continuous predictions on real-time data streams. It's useful for applications that require continuous monitoring and immediate action based on predictions.

### Chosen Implementation

For this project, the **Real-time Prediction** pattern was chosen. This approach provides the following benefits:

- **Immediate Response**: Users can get instant predictions for specific items, making it interactive and user-friendly.
- **Scalability**: The model can be easily scaled to handle more requests by deploying it in a containerized environment like Docker.
- **Flexibility**: The model can be updated and redeployed without affecting the application, ensuring continuous improvement and maintenance.

### Setup

#### Step 1: Install Dependencies

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Step 2: Run the Flask Application

You can run the Flask application either locally or using Docker.

##### Running Locally

```bash
python scripts/app.py
```

The Flask application will be available at `http://localhost:5000`.

##### Running with Docker

Build the Docker image:

```bash
docker build -t sales-forecasting-app .
```

Run the Docker container:

```bash
docker run -p 5000:5000 sales-forecasting-app
```

The Flask application will be available at `http://localhost:5000`.

### API Endpoint

#### `/predict`

**Method**: `GET`

**Description**: Predict the sales for a given item in all shops for the month of November 2015.

**Parameters**:
- `item_id` (int): The ID of the item to predict sales for.

**Response**:
- `item_id` (int): The ID of the item.
- `predictions` (list): A list of dictionaries containing the `shop_id` and the `predicted_sales` for each shop.

**Example Request**:
```bash
curl -X GET "http://localhost:5000/predict?item_id=5678"
```

**Example Response**:
```json
{
  "item_id": 5678, 
  "predictions": [
    {
      "predicted_sales": 2, 
      "shop_id": 3
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 4
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 5
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 6
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 7
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 9
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 10
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 12
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 15
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 16
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 18
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 19
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 21
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 22
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 24
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 25
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 26
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 27
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 28
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 29
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 30
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 31
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 34
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 35
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 37
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 38
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 39
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 42
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 43
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 44
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 46
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 47
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 48
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 50
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 52
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 53
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 54
    }, 
    {
      "predicted_sales": 2, 
      "shop_id": 56
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 57
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 58
    }, 
    {
      "predicted_sales": 3, 
      "shop_id": 59
    }
  ]
}
```

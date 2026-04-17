import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Load the dataset
# Make sure 'train.csv' is inside the 'data' folder
df = pd.read_csv("data/train.csv")

# STEP 2: Select input features (as per task)
# GrLivArea = Area, BedroomAbvGr = Bedrooms, FullBath = Bathrooms
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Target variable (what we want to predict)
y = df['SalePrice']

# STEP 3: Split dataset into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 5: Make predictions on test data
y_pred = model.predict(X_test)

# STEP 6: Evaluate model performance

# MAE: Average prediction error
mae = mean_absolute_error(y_test, y_pred)

# MSE: Squared error (penalizes larger errors more)
mse = mean_squared_error(y_test, y_pred)

# R2 Score: Measures how well the model fits the data (closer to 1 is better)
r2 = r2_score(y_test, y_pred)

# Print performance results
print("Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# STEP 7: Test the model with a sample input
# Format: [Area, Bedrooms, Bathrooms]
sample = [[2000, 3, 2]]

# Predict house price for sample input
prediction = model.predict(sample)

# Print predicted price
print(f"\nPredicted House Price: {prediction[0]:.2f}")
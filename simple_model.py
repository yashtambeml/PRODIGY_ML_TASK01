import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/train.csv")

# Features and target
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# ✅ FIXED SAMPLE PREDICTION (NO WARNING)
sample = pd.DataFrame([[2000, 3, 2]],
                      columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])

prediction = model.predict(sample)
print(f"\nPredicted House Price: {prediction[0]:.2f}")

# Graph 1: Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

# perfect line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.show()

# Graph 2: Square Footage vs Price
plt.figure(figsize=(8,6))
plt.scatter(X_test['GrLivArea'], y_test, label="Actual", alpha=0.5)
plt.scatter(X_test['GrLivArea'], y_pred, label="Predicted", alpha=0.5)

plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("House Size vs Price")
plt.legend()

plt.show()
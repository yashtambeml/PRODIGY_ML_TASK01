import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Load dataset
df = pd.read_csv("data/train.csv")

# STEP 2: Features and target
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# STEP 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Model training
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 5: Predictions
y_pred = model.predict(X_test)

# STEP 6: Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# STEP 7: Sample prediction
sample = [[2000, 3, 2]]
pred = model.predict(sample)
print(f"\nPredicted Price for sample house: {pred[0]:.2f}")

# STEP 8: GRAPH 1 - Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

# perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.show()

# STEP 9: GRAPH 2 - Square Footage vs Price (Visual Insight)
plt.figure(figsize=(8,6))
plt.scatter(X_test['GrLivArea'], y_test, label="Actual Price", alpha=0.5)
plt.scatter(X_test['GrLivArea'], y_pred, label="Predicted Price", alpha=0.5)

plt.xlabel("Square Footage (GrLivArea)")
plt.ylabel("House Price")
plt.title("Square Footage vs House Price")
plt.legend()

plt.show()
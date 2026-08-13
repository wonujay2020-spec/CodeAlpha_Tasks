import pandas as pd
df = pd.read_csv("car data.csv")
print(df.head())
print(df.columns)
print(df.dtypes)
print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

X = df[['Year', 'Present_Price', 'Driven_kms', 'Owner']]
y = df['Selling_Price']

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape) 

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

print("\nPredicted Prices:")
print(y_pred[:5])

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

sample_car = pd.DataFrame({
    'Year': [2015],
    'Present_Price': [5.0],
    'Driven_kms': [30000],
    'Owner': [0]
})

sample_prediction = model.predict(sample_car)

print("\nSample Car Price Prediction:")
print(sample_prediction[0])

import matplotlib.pyplot as plt

# Actual vs Predicted Prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()


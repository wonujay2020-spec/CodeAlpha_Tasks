# Car Price Prediction

## Project Overview

This project uses machine learning to predict the selling price of used cars based on features such as year, present price, kilometers driven, and number of previous owners.

The project was completed as part of the CodeAlpha Data Science Internship.

## Dataset

The dataset contains information about used cars, including:

- Year
- Present Price
- Driven Kilometers
- Owner
- Selling Price

There are 301 records and 9 columns in the dataset.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Methodology

The following steps were carried out:

1. Loaded the car dataset using Pandas.
2. Inspected the dataset and its data types.
3. Checked for missing values.
4. Selected relevant features for prediction.
5. Split the dataset into training and testing sets.
6. Built a Linear Regression model.
7. Trained the model using the training data.
8. Generated predictions using the test data.
9. Evaluated the model using MAE, MSE, and R² Score.
10. Created a visualization comparing actual and predicted selling prices.

## Model Performance

The Linear Regression model achieved an R² Score of approximately 0.82 on the test data.

The evaluation results were:

- Mean Absolute Error: approximately 1.30
- Mean Squared Error: approximately 4.13
- R² Score: approximately 0.82

## Sample Prediction

The model was also used to predict the selling price of a sample car based on its features.

The predicted price was approximately 3.97.

## Visualization

An Actual vs Predicted Selling Price scatter plot was created to visualize the model's predictions.

## Author

CodeAlpha Data Science Intern
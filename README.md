# Predicting Movie Rental Duration

## Project Overview

A DVD rental company wants to improve its inventory planning by predicting how many days customers will keep rented DVDs.

This project applies supervised machine learning regression techniques to estimate rental duration based on movie characteristics, rental information, and DVD attributes.

The main objective is to identify a model achieving a Mean Squared Error (MSE) of 3 or less on the test dataset.

This project was completed as part of a DataCamp guided project.

## Business Problem

Accurately estimating DVD rental durations can help a rental company anticipate DVD availability and improve inventory management.

The challenge is to develop a predictive model that estimates the number of rental days using historical rental data.

## Dataset

The project uses `rental_info.csv`.

The dataset contains information about:

- Rental dates and return dates.
- Rental amounts and rental rates.
- Movie release years and durations.
- DVD replacement costs.
- Special movie features.
- Movie rating categories.

### Target Variable

`rental_length_days`: the number of days between the rental date and return date.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn

## Methodology

### 1. Data Preparation and Feature Engineering

- Convert rental and return dates into datetime objects.
- Calculate rental duration in days.
- Extract binary features from special DVD features.
- Separate explanatory variables from the target variable.
- Split the dataset into training and testing sets (80/20).

### 2. Feature Selection

Lasso Regression is used to identify relevant features by shrinking less useful coefficients toward zero.

The selected features are subsequently used to train a Linear Regression model.

### 3. Model Development

Two regression approaches are investigated:

- Linear Regression: a baseline linear modeling approach using selected features.
- Random Forest Regressor: an ensemble learning algorithm capable of capturing nonlinear relationships.

### 4. Hyperparameter Optimization

RandomizedSearchCV with 5-fold cross-validation is used to optimize Random Forest hyperparameters.

The search explores:

- `n_estimators`: number of decision trees.
- `max_depth`: maximum depth of each tree.

### 5. Model Evaluation

The models are evaluated using:

- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).

The main business requirement is:

**Test MSE ≤ 3**

## Results

The final comparison includes:

| Model | Test MSE | Test RMSE |
|---|---:|---:|
| Linear Regression | To be added | To be added |
| Random Forest | To be added | To be added |

The final model is selected based on its test performance and the business requirement.

## Key Takeaways

This project demonstrates how supervised machine learning can be applied to rental duration prediction.

It also explores the importance of feature engineering, feature selection, cross-validation, and hyperparameter tuning when developing regression models.

## Project Structure

```text
movie-rental-duration/
├── README.md
├── notebook.ipynb
├── rental_info.csv
├── requirements.txt
└── .gitignore
```

## How to Run

1. Clone the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Place `rental_info.csv` in the `data/` directory.
4. Open the notebook and run the cells.

## Limitations

- The analysis relies on historical rental information.
- The model does not account for all possible external factors influencing rental duration.
- The prediction performance depends on the quality and representativeness of the available dataset.

## Acknowledgements

This project was completed as part of a DataCamp guided project.


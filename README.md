

 Boston House Price Prediction

Overview

The Boston House Price Prediction project aims to predict the median price of houses in Boston based on various features like crime rate, number of rooms, proximity to employment centers, and more. Using the famous **Boston Housing Dataset**, this project applies various machine learning models to analyze and predict housing prices. The goal is to develop a predictive model that can help estimate property values, which can be useful for buyers, sellers, and real estate agencies.

Dataset

The dataset used in this project is the **Boston Housing Dataset**, which contains 506 instances and 14 attributes. The features include both continuous and categorical data that impact housing prices in the Boston area.

Key features in the dataset:
- CRIM: Per capita crime rate by town
- ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX: Nitric oxides concentration (parts per 10 million)
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built before 1940
- DIS: Weighted distances to five Boston employment centers
- RAD: Index of accessibility to radial highways
- TAX: Full-value property tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
- LSTAT: Percentage of lower status of the population
- MEDV: Median value of owner-occupied homes in $1000's (Target Variable)

 Project Workflow

1. Data Preprocessing: 
    - Handle missing values, outliers, and data scaling.
    - Feature selection and engineering to improve model performance.

2. Exploratory Data Analysis (EDA):
    - Visualize data distribution, correlation between features, and relationships between features and the target variable.

3. Modeling:
    - Implement various regression models such as:
        - Linear Regression
        - Decision Tree Regression
        - Random Forest Regression
        - Gradient Boosting Regression
        - XGBoost
        
4. Model Evaluation:
    - Evaluate models based on performance metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared score.
    - Cross-validation and hyperparameter tuning for improved accuracy.

5. Prediction:
    - Use the best-performing model to predict house prices on the test dataset.

Requirements

- Python 3.x
- Libraries: 
    - `numpy`
    - `pandas`
    - `matplotlib`
    - `seaborn`
    - `scikit-learn`
    - `xgboost`

Results

- A detailed report on model performance and predictions.
- Visualization of feature importance and the relationship between features and house prices.

Conclusion

This project demonstrates how machine learning techniques can be applied to predict house prices based on a variety of factors. By comparing different regression models, the best-performing model can be used to accurately estimate real estate values in the Boston area.



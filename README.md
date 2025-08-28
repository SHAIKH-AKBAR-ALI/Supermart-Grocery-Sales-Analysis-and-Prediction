Supermart Grocery Sales Analysis and Prediction
Overview
This repository contains a Jupyter Notebook (Supermart-Grocery-Sales-Analysis-and-Prediction.ipynb) that analyzes the "Supermart Grocery Sales - Retail Analytics Dataset" to derive insights and predict sales using a Linear Regression model. The notebook covers data preprocessing, exploratory data analysis (EDA), feature engineering, and model evaluation.
Dataset
The dataset (Supermart Grocery Sales - Retail Analytics Dataset.csv) includes grocery sales data with columns like Order ID, Customer Name, Category, Sub Category, City, Order Date, Region, Sales, Discount, Profit, and State.
Notebook Contents

Data Preprocessing:
Load dataset using Pandas.
Check for missing values and duplicates.
Convert categorical variables (Category, Sub Category, City, Region, State) to numerical using LabelEncoder.


Exploratory Data Analysis (EDA):
Summary statistics and data distribution visualization (e.g., Kernel Density Estimation plot for Sales).
Category-wise sales distribution using value counts.


Feature Engineering:
Drop non-predictive columns (e.g., Order ID, Customer Name, Order Date).
Standardize numerical features using StandardScaler.


Model Building:
Split data into training and testing sets (80-20 split).
Train a Linear Regression model to predict Sales.
Evaluate model performance using Mean Squared Error (MSE) and R-Squared metrics.


Libraries Used:
Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.



Requirements
To run the notebook, install the required Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

How to Run

Clone this repository:git clone https://github.com/your-username/your-repo-name.git


Ensure the dataset (Supermart Grocery Sales - Retail Analytics Dataset.csv) is in the same directory as the notebook.
Open the notebook in Jupyter:jupyter notebook Supermart-Grocery-Sales-Analysis-and-Prediction.ipynb


Run all cells to execute the analysis and model training.

Results

The notebook provides insights into sales patterns across categories and regions.
The Linear Regression model predicts sales with an MSE of ~212,757 and R-Squared of ~0.355, indicating moderate predictive performance.

Usage
This project is ideal for data science enthusiasts, retail analysts, or anyone interested in applying machine learning to sales data. Modify the notebook to experiment with other models (e.g., Random Forest, XGBoost) or additional features for improved predictions.
License
This project is licensed under the MIT License. See the LICENSE file for details.

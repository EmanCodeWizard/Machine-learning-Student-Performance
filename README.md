# Machine-learning-Student-Performance
1. Data Preprocessing and EDA (Exploratory Data Analysis)
1.1 Import Libraries
Essential Python libraries are imported:

pandas, numpy for data handling

matplotlib, seaborn for visualization

scikit-learn for machine learning models and metrics

scipy.stats for statistical plots (Q-Q plot)

1.2 Load Dataset & Preview
Student_Performance.csv is loaded into a DataFrame called df.

df.head() shows the first few rows to understand the data structure.

1.3 Check for Missing Values
df.isnull().sum() checks for missing data in each column.

1.4 Convert Categorical Variables
Extracurricular Activities is categorical (Yes/No), so it's encoded as:

Yes → 1

No → 0

1.5 Feature Scaling
StandardScaler standardizes all features (mean = 0, std = 1).

A new DataFrame scaled_df is created from the scaled values.

1.6 EDA - Visual Analysis
Distribution Plot: Visualizes how Performance Index is distributed.

Scatter Plots: Show how each feature (like Hours Studied, Sleep Hours, etc.) correlates with the Performance Index.

Correlation Heatmap: Displays correlation coefficients between all features and the target variable.

2. Linear Regression Model Implementation
2.1 Simple Linear Regression
Predict Performance Index using only Hours Studied.

Train-test split (80-20), model training, and visualization of the regression line.

2.2 Multiple Linear Regression
Uses all features to predict Performance Index.

Model is trained, and coefficients (impact of each feature) are displayed.

3. Model Evaluation
3.1 Metrics Used
MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

R² Score (coefficient of determination)

Adjusted R²: Better than R² when multiple predictors are used (penalizes non-useful variables).

Both Simple and Multiple Linear Regression models are evaluated using these metrics.

4. Residual Analysis
Checks if residuals (errors between actual and predicted values) are random and normally distributed:

Scatter Plot: Should show no clear pattern.

Histogram + KDE: Should resemble a bell curve.

Q-Q Plot: Should lie roughly on a straight line if residuals are normally distributed.

This helps validate assumptions of linear regression.

5. Hyperparameter Tuning and Model Improvement
5.1 Ridge and Lasso Regression
Ridge Regression: Adds penalty on coefficients to prevent overfitting.

Lasso Regression: Like Ridge, but also performs feature selection by shrinking some coefficients to zero.

Both use cross-validation to select the best penalty parameter (alpha).

Evaluation
Predict using both Ridge and Lasso models.

Compare RMSE values for model quality.

5.2 Feature Importance via Lasso
The most influential features are identified from Lasso coefficients.

Top 3 features with the highest absolute coefficients are selected.

A new Linear Regression model is trained using only those top 3 features.

RMSE is calculated again to evaluate performance improvement.


# Fraud-Dectection-Model
# Project Introduction
This project focuses on the critical domain of fraud detection within transactional data. Specifically, my objective is to leverage various transaction and user attributes to predict the fraud rate for individual users. This is based on the same dataset as the fraud detection dashboard

The overall goal of this notebook is to develop and evaluate machine learning models capable of accurately forecasting these fraud rates, thereby enhancing security measures and potentially mitigating financial losses.

# Data Loading and Initial Exploration
To begin our analysis, the dataset was loaded from the file 'Fraud Detection Dataset.csv' into a pandas DataFrame, which we named data. During this loading process, the 'Transaction_ID' column was designated as the index for the DataFrame, providing a unique identifier for each transaction.

Following the data loading, initial exploration was performed using two key pandas functions:

data.head(10): This command was executed to display the first 10 rows of the DataFrame. This provided an immediate visual overview of the dataset's structure, column names, and the types of values contained within, helping us to quickly grasp the nature of the transactional data.

data.info(): This function was then used to obtain a concise summary of the DataFrame. The output included the number of entries, the total number of columns, the non-null count for each column, and their respective data types. Critically, the non-null counts helped in swiftly identifying which columns contained missing values, setting the stage for subsequent data preprocessing steps.

# Data Preprocessing Pipeline Explained:
Our data preprocessing pipeline involved several crucial steps to prepare the dataset for machine learning models:

1. Handling Missing Values: We began by addressing missing values in the dataset. For categorical columns (identified by dtype == 'object'), missing entries were imputed with the mode (most frequent value) of their respective columns. For numerical columns (identified by dtype as int64 or float64), missing entries were filled with the mean of their columns. This ensures that no data points are lost and the dataset is complete. After imputation, a verification step data.isnull().sum() confirmed that there were no remaining missing values across any columns, making the data robust for further analysis.

2. Creating the fraud_rate Target Variable: Initially, the dataset contained a binary Fraudulent column (0 or 1). To pivot towards a regression task, we engineered a new target variable called fraud_rate. This was achieved by grouping the data by User_ID and calculating the mean of the Fraudulent column for each user. This fraud_rate (representing the proportion of fraudulent transactions for a given user) was then merged back into our main DataFrame, assigning each transaction its associated user's fraud rate. Subsequently, the original Fraudulent column was dropped as fraud_rate now serves as our continuous target variable.

3. Converting Categorical Features to Dummy Variables (One-Hot Encoding): Machine learning models typically require numerical input. Therefore, all categorical columns (e.g., Transaction_Type, Device_Used, Location, Payment_Method) were converted into numerical representations using one-hot encoding via pd.get_dummies(). To prevent multicollinearity, which can negatively impact model performance, we applied drop_first=True. This drops the first category in each one-hot encoded set, reducing redundancy and making the model more stable.

3. Standardizing Numerical Features: Finally, to ensure that all numerical features contribute equally to the model training process and to prevent features with larger scales from dominating those with smaller scales, we performed feature standardization. The target variable (fraud_rate) and the User_ID were separated from the rest of the features. A StandardScaler was then applied to the remaining feature set (x), transforming them to have a mean of 0 and a standard deviation of 1. This results in our scaled feature matrix, x_scaled, which is ready for model training.

# Exploratory Data Analysis (EDA)

Correlation Analysis
To understand the relationships between our features and the target variable, fraud_rate, we performed a comprehensive correlation analysis. The purpose of this analysis is to identify which features have a strong, weak, or negligible linear relationship with the fraud_rate, and also to detect potential multicollinearity among predictor variables.

Calculation of Correlation Matrix
We calculated the full correlation matrix for our dataset using data.corr(). This matrix provides Pearson correlation coefficients between all pairs of variables. To specifically examine the relationship of each feature with our target variable, we then extracted and sorted the correlations with fraud_rate using correlation_matrix['fraud_rate'].sort_values(ascending=False).

Heatmap Visualization
To visually represent the entire correlation matrix, we generated a heatmap using sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f"). This visualization allowed us to quickly identify strong positive and negative correlations, as well as to observe the relationships between all predictor variables themselves. It helped in understanding both feature-target and feature-feature relationships, which is crucial for model building.

Scatter Plot for 'Previous_Fraudulent_Transactions' vs. 'fraud_rate'
Given that Previous_Fraudulent_Transactions showed some relationship with the fraud_rate, we further explored this connection with a scatter plot: sns.scatterplot(x='Previous_Fraudulent_Transactions', y='fraud_rate', data=data). This plot was intended to visualize the nature of this relationship. From the scatter plot, it was observed that the relationship between Previous_Fraudulent_Transactions and fraud_rate appeared to be non-linear. This observation was a key insight, suggesting that models capable of capturing non-linear patterns, such as Decision Trees or Random Forests, would be more appropriate for this prediction task.

# Model Development and Training
Subtask:
Outline the process of splitting the data into training and testing sets. Detail the implementation and training of both the Decision Tree Regressor and the Random Forest Regressor models.

Data Splitting:
The dataset was divided into training and testing sets to evaluate model performance on unseen data. The features (x_scaled), which are the standardized numerical and one-hot encoded categorical variables, and the target variable (y), which represents the calculated fraud_rate, were used for this split. Specifically, 20% of the data was allocated to the testing set (test_size=0.20), ensuring a random_state of 1 for reproducibility and shuffle=False to maintain the original order of the data.

Decision Tree Regressor Implementation:
To address the potentially non-linear relationship observed in the data, a Decision Tree Regressor was implemented. The DecisionTreeRegressor model was instantiated with a max_depth of 5 to control complexity and prevent overfitting, and a random_state of 42 for consistent results. This model was then trained using the prepared training features (X_train) and their corresponding target values (y_train).

Random Forest Regressor Implementation:
Following the Decision Tree, a Random Forest Regressor, an ensemble method known for its robustness and improved accuracy, was also implemented. The RandomForestRegressor model was instantiated with a random_state of 42 for reproducibility. This ensemble model was then fitted to the same training data (X_train, y_train) to make predictions on the fraud rate.

# Model Evaluation
To evaluate the performance of our regression models (Decision Tree Regressor and Random Forest Regressor), we utilized two key metrics:

Mean Squared Error (MSE): Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. A lower MSE indicates a better fit of the model to the data.
R-squared (R2) Score: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. R2 scores range from 0 to 1, with 1 indicating a perfect fit. A negative R2 score, however, means the model is performing worse than simply predicting the mean of the target variable.
Decision Tree Regressor Evaluation
MSE: 0.003776694066292765
R2 Score: -0.006516882718893058
The negative R2 score for the Decision Tree Regressor indicates that this model performs worse than a simple horizontal line at the mean of fraud_rate. This suggests that the Decision Tree model, with its current configuration, is not able to capture any meaningful patterns in the data.

Random Forest Regressor Evaluation
MSE: 0.0036022991111769585
R2 Score: 0.03996066179594926
The Random Forest Regressor showed a slight improvement with a positive R2 score of approximately 0.0399. While positive, this score is still very low, indicating that the model explains only about 4% of the variance in the fraud_rate. This suggests that while the Random Forest model is marginally better than the Decision Tree and better than simply predicting the mean, its predictive power with the current feature set is quite limited. This aligns with the earlier observation of weak correlations between features and the target variable.

# Key Findings and Challenges
Summary of Model Evaluations
Decision Tree Regressor:

The Decision Tree Regressor model yielded a negative R2 score (-0.0065). This indicates that the model performed worse than simply predicting the mean of the target variable (fraud_rate), suggesting a very poor fit and an inability to capture any meaningful patterns in the data.
Random Forest Regressor:

The Random Forest Regressor showed a slight improvement with a positive R2 score of approximately 0.0399.
While positive, this R2 score is still very low, explaining only about 4% of the variance in the fraud_rate. This means that the model's predictive power is extremely limited, and a large proportion of the variability in fraud rates remains unexplained by the current features.
Challenges and Insights
Weak Feature-Target Relationship: The consistently low R2 scores across both models, combined with the initial correlation analysis, strongly suggest a weak relationship between the existing features in the dataset and the target fraud_rate. The predictive variables do not seem to have a strong explanatory power over how frequently fraud occurs for a user.
Complexity of Fraud Rate Prediction: Predicting user-specific fraud rates is inherently complex. The models struggled to generalize and make accurate predictions, indicating that the factors driving fraud rates might be more intricate than what the current dataset captures.
Limited Predictive Power: The models, even the more robust Random Forest, provide very limited predictive power. This implies that relying on these models alone for critical fraud detection decisions would be ineffective.
Need for More Influential Features: The results highlight a critical need for either more advanced feature engineering from existing data or the integration of additional, more influential data sources to better explain the variance in fraud_rate.

# Data Analysis Key Findings
Data Loading and Initial Exploration: The dataset was loaded from 'Fraud Detection Dataset.csv', using 'Transaction_ID' as the index. Initial exploration through data.head(10) and data.info() helped to understand the data structure and identify missing values, which were subsequently addressed.

Data Preprocessing:
Missing values were handled by imputing categorical columns with their mode and numerical columns with their mean.
A new target variable, fraud_rate, was created by calculating the mean of the Fraudulent column grouped by User_ID, transforming the problem into a regression task. The original Fraudulent column was then dropped.

Categorical features were converted into numerical dummy variables using one-hot encoding with drop_first=True to prevent multicollinearity.
Numerical features were standardized using StandardScaler to ensure all features contributed equally to model training.

Exploratory Data Analysis (EDA):
Correlation analysis was performed, including the computation of a correlation matrix and specific correlations with the fraud_rate target.
A heatmap was used for a visual representation of the correlation matrix, and a scatter plot of 'Previous_Fraudulent_Transactions' vs. fraud_rate indicated a non-linear relationship.

Model Performance:
The Decision Tree Regressor yielded a Mean Squared Error (MSE) of approximately 0.003777 and an R-squared (R2) score of -0.006517. The negative R2 score indicates that this model performs worse than simply predicting the mean fraud_rate.
The Random Forest Regressor showed a slight improvement with an MSE of approximately 0.003602 and an R2 score of 0.039961. While positive, this R2 score is very low, explaining only about 4% of the variance in the fraud_rate.

Challenges Identified: The consistently low R2 scores across both models, combined with initial correlation analysis, suggest a weak relationship between the existing features and the target fraud_rate, indicating limited predictive power from the current dataset.

Insights or Next Steps
Enhance Feature Engineering: Given the limited predictive power of current models, the next steps should focus on advanced feature engineering, including exploring interaction effects, temporal patterns (e.g., transaction frequency over time), and deriving new domain-specific features that might better capture complex fraud behaviors.
Explore Alternative Data and Models: Integrate external data sources such as user behavior logs, device intelligence, or geographical risk factors. Simultaneously, investigate more sophisticated machine learning algorithms like Gradient Boosting models (XGBoost, LightGBM, CatBoost) or deep learning techniques, which are better equipped to handle complex, non-linear relationships and potentially improve model performance significantly.


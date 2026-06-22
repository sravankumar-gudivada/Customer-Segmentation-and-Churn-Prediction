# Customer Segmentation and Churn Prediction

This project analyzes telecom customer churn and builds machine learning models to identify customers who are likely to leave. It also segments customers into actionable groups using predicted churn probability, tenure, monthly charges, and total charges.

The work is implemented in [`ml_model_1.ipynb`](ml_model_1.ipynb) using the Telco customer churn dataset stored in [`Telco_customer_churn.xlsx`](Telco_customer_churn.xlsx).

## Project Overview

Customer churn is a critical business problem for subscription-based telecom companies. The goal of this project is to:

- Explore customer behavior and churn patterns.
- Clean and prepare the dataset for modeling.
- Train Random Forest classification models to predict churn.
- Improve churn detection using class balancing and hyperparameter tuning.
- Identify important features that influence churn.
- Segment customers into business-friendly groups for targeted retention strategies.

## Dataset

The dataset contains 7,043 customer records and 33 variables covering demographic information, location, subscribed services, billing details, churn status, churn score, customer lifetime value, and churn reason.

Key target fields:

- `Churn Label`: Indicates whether the customer churned, with values `Yes` or `No`.
- `Churn Value`: Binary target variable where `1` means the customer churned and `0` means the customer stayed.

Important feature groups include:

- Customer profile: `Gender`, `Senior Citizen`, `Partner`, `Dependents`
- Service usage: `Phone Service`, `Internet Service`, `Online Security`, `Tech Support`, `Streaming TV`, `Streaming Movies`
- Contract and billing: `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges`, `Total Charges`
- Customer value and churn metadata: `Churn Score`, `CLTV`, `Churn Reason`

The class distribution is imbalanced:

| Churn Value | Customer Count |
|---|---:|
| 0 - Not Churned | 5,174 |
| 1 - Churned | 1,869 |

## Notebook Workflow

### 1. Exploratory Data Analysis

The notebook investigates churn patterns across tenure, monthly charges, contracts, internet service, payment method, and tech support.

Notable observations from the analysis:

- Churned customers have a much lower average tenure than retained customers.
- Month-to-month contracts show a much higher churn rate than one-year and two-year contracts.
- Customers with higher monthly charges are more represented among churned customers.
- Service and support-related features such as internet service type and tech support are important churn indicators.

### 2. Data Cleaning and Preparation

The preprocessing steps include:

- Converting `Total Charges` from object type to numeric.
- Filling missing `Total Charges` values with `0` for customers with zero tenure.
- Removing columns that are identifiers, location metadata, direct churn explanations, or unsuitable for model training.
- Dropping high-cardinality `City` before encoding.
- Removing `Churn Label` and using `Churn Value` as the target.
- Applying one-hot encoding to categorical variables.

After preprocessing, the final encoded modeling dataset contains 31 columns.

### 3. Churn Prediction Modeling

The notebook trains and evaluates several Random Forest models:

- Baseline Random Forest
- Class-balanced Random Forest
- Tuned Random Forest with `n_estimators=300`, `max_depth=10`, and `class_weight="balanced"`
- Feature-selected Random Forest
- Grid-style comparison across tree counts and maximum depths

The tuned Random Forest is the strongest practical model for churn detection because it improves recall for churned customers.

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1-score |
|---|---:|---:|---:|---:|
| Baseline Random Forest | 0.7857 | 0.66 | 0.51 | 0.58 |
| Balanced Random Forest | 0.7921 | 0.67 | 0.52 | 0.59 |
| Tuned Random Forest | 0.7828 | 0.59 | 0.75 | 0.66 |
| Selected Features Random Forest | 0.7835 | 0.60 | 0.74 | 0.66 |

Additional validation results for the tuned model:

- Mean 5-fold cross-validation accuracy: `0.7794`
- Mean 5-fold cross-validation recall: `0.7335`
- ROC-AUC score: `0.8571`

### 4. Feature Importance

The most influential features in the baseline Random Forest include:

1. `Total Charges`
2. `Tenure Months`
3. `Monthly Charges`
4. `Payment Method_Electronic check`
5. `Dependents_Yes`
6. `Internet Service_Fiber optic`
7. `Contract_Two year`
8. `Gender_Male`
9. `Partner_Yes`
10. `Paperless Billing_Yes`

These features suggest that customer tenure, billing amount, payment behavior, contract type, and service profile are strong indicators of churn risk.

### 5. Customer Segmentation

After churn prediction, the notebook uses the tuned Random Forest model to estimate churn probability for every customer. These probabilities are combined with:

- `Tenure Months`
- `Monthly Charges`
- `Total Charges`
- `Churn Probability`

The features are standardized and clustered using K-Means. The elbow method is used to inspect the number of clusters, and the notebook creates three customer segments:

| Cluster | Segment Name | Average Tenure | Average Monthly Charges | Average Total Charges | Average Churn Probability |
|---:|---|---:|---:|---:|---:|
| 0 | Budget Loyal Customers | 32.05 | 32.85 | 1,047.70 | 0.1206 |
| 1 | High Risk New Customers | 10.96 | 71.96 | 884.07 | 0.6914 |
| 2 | Premium Loyal Customers | 58.40 | 90.43 | 5,278.00 | 0.2306 |

These segments can support targeted business actions:

- High Risk New Customers: Prioritize retention offers, onboarding support, and service issue resolution.
- Budget Loyal Customers: Maintain satisfaction with low-cost loyalty benefits.
- Premium Loyal Customers: Protect high-value accounts with proactive service and personalized engagement.

## Project Structure

```text
.
|-- ml_model_1.ipynb
|-- Telco_customer_churn.xlsx
`-- README.md
```

## Requirements

Install the following Python libraries before running the notebook:

```bash
pip install -r requirements.txt
```

## How to Run

1. Open `ml_model_1.ipynb` in Jupyter Notebook, JupyterLab, Google Colab, or VS Code.
2. Make sure `Telco_customer_churn.xlsx` is available in the project directory.
3. Update the dataset path if needed:

```python
df = pd.read_excel("Telco_customer_churn.xlsx")
```

4. Run the notebook cells from top to bottom.

## Business Value

This project helps telecom teams identify customers most likely to churn and understand the factors associated with churn risk. The segmentation layer makes the model output easier to act on by grouping customers into interpretable business categories, allowing teams to design more focused retention campaigns.

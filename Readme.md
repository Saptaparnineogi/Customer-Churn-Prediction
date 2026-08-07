# Customer Churn Risk Prediction with Model Explanability
An end-to-end machine learning project that predicts customer churn and demonstrates how predictive analytics can support business-driven retention strategies.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-End--to--End-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


![Churn by Contract](figures/workflow.png)
>The figure above summarizes the end-to-end machine learning workflow implemented in this project.

## Executive Summary

The project demonstrates the complete lifecycle of a production-style machine learning solution, from exploratory analysis and model development to automated inference on new customer data and business decision support.

Highlights include:

- Built an end-to-end production-style machine learning pipeline
- Benchmarked four classification algorithms
- Automated preprocessing using sklearn Pipelines
- Optimized the decision threshold for business-oriented classification
- Persisted trained models and metadata for reproducible inference
- Developed a standalone prediction pipeline for new customer data
- Added unit tests and GitHub Actions CI

The project demonstrates the complete lifecycle of a production-style machine learning solution, from exploratory analysis to automated inference on new customer data., from exploratory analysis to business decision support.

## Project Overview

Customer churn is one of the most important challenges faced by subscription-based businesses. Acquiring new customers is significantly more expensive than retaining existing ones, making early identification of customers at risk of leaving a valuable business capability.

This project develops an end-to-end machine learning pipeline to predict customer churn using the IBM Telco Customer Churn dataset. Beyond predictive modelling, the project focuses on business interpretation by incorporating:

- Business-driven feature engineering
- Model benchmarking
- Threshold optimisation
- Business impact simulation
- SHAP explainability

The objective is not only to predict churn accurately but also to demonstrate how machine learning can support customer retention strategies in a real business environment.

## Project Components

- Exploratory Data Analysis (EDA)
- Business-driven Feature Engineering
- Multiple ML Algorithms
  - Logistic Regression
  - Random Forest
  - XGBoost
  - CatBoost
- Model Benchmarking
- Threshold Optimisation
- Business Impact Simulation
- SHAP Explainability
- Batch Prediction Pipeline

## Technologies

**Languages & Libraries**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-017CEE?style=flat)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-green)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Model Performance


| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---------------------|---------:|----------:|-------:|---------:|--------:|-------:|
| Random Forest       | 0.7573   | 0.5287    | **0.7888** | **0.6330** | **0.8445** | 0.6521 |
| Logistic Regression | 0.7317   | 0.4966    | 0.7861 | 0.6087 | 0.8420 | 0.6357 |
| CatBoost            | **0.8020** | **0.6578** | 0.5294 | 0.5867 | 0.8432 | **0.6565** |
| XGBoost             | 0.8006   | 0.6545    | 0.5267 | 0.5837 | 0.8410 | 0.6514 |

>**Selected model: Random Forest, chosen because it achieved the highest F1 score and strong recall, which better supports identifying customers at risk of churn.**

## Decision Threshold Optimization

![Decision Threshold Optimization](figures/threshold_optimisation.png)

The default classification threshold of **0.50** was compared with alternative thresholds using precision, recall and F1-score. A threshold of **0.54** produced the strongest F1-score, improving the balance between identifying customers likely to churn and avoiding unnecessary retention interventions.

At the selected threshold:

- **Precision:** 0.554
- **Recall:** 0.757
- **F1-score:** 0.640

This threshold was therefore used to convert predicted churn probabilities into customer classifications for the subsequent risk-segmentation and business analysis.

## 📈 Key Results & Visualizations

### 1. Key Exploratory Finding

![Churn by Contract](figures/churn_by_contract.png)

Month-to-month customers exhibit substantially higher churn than customers on one-year or two-year contracts, establishing contract commitment as one of the clearest retention indicators.

### 2. Model Benchmarking
![Churn by Contract](figures/model_comparison.png)

Random Forest was selected because it maintained high recall while achieving the highest F1-score, providing the best balance between detecting churners and limiting unnecessary retention interventions.

### 3. What Drives Customer Churn?
![Churn by Contract](figures/feature_importance.png)
Contract type, customer tenure, pricing, internet service, and access to support services emerged as the strongest predictors. Month-to-month customers and customers early in their lifecycle represent particularly important retention segments.

### 4. Probability-Based Customer Risk Segmentation
![Churn by Contract](figures/risk_segmentation.png)

The model successfully separates customers into meaningful risk groups. Observed churn rises from **5.7%** in the Low-Risk segment to **74.2%** in the Very-High-Risk segment, enabling retention teams to prioritise resources according to customer risk.

### 5. Explainable AI with SHAP

![SHAP summary](figures/shap_summary.png)



## Business Insights

The analysis demonstrates how churn predictions can support customer retention strategies by enabling:

- Early identification of high-risk customers
- Prioritised allocation of retention resources
- Explainable prediction using SHAP
- Data-driven retention decision making


## Repository Structure

```text
Customer-Churn-Prediction/
│
├──.github/
|   └── workflows/
      └── tests.yml
├── data/
│   ├── raw/
│   └── processed/
│
├── figures/
│   ├── churn_by_contract.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── risk_segmentation.png
│   └── shap_summary.png
│
├── models/
│   ├── best_model.joblib
│
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Feature_Engineering_and_Model_Development.ipynb
│   └── 03_Model_Evaluation_and_Business_Insights.ipynb
│── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── decision_threshold.py
│   └── model_io.py
│
├── tests/
|   ├── test_decision_threshold.py
|   ├── test_model_io.py
|   └── test_preprocessing.py
|
├── train.py
├── predict.py
├── README.md
└── requirements.txt

```
> The repository is organised to separate exploratory analysis, model development, trained models and supporting visualisations, making the workflow easy to follow and reproduce.

## Notebook Guide

| Notebook | Purpose |
|----------|---------|
| `01_Exploratory_Data_Analysis.ipynb` | Explores churn patterns across customer demographics, contracts, services, tenure and billing behaviour |
| `02_Feature_Engineering_and_Model_Development.ipynb` | Builds business-driven features, preprocessing pipelines and benchmarks multiple classification models |
| `03_Model_Evaluation_and_Business_Insights.ipynb` | Optimises the decision threshold, segments customers by risk, simulates business impact and explains predictions using SHAP |

## Dataset

The project uses the IBM Telco Customer Churn dataset, containing customer demographics, account information, subscribed services, billing characteristics and churn outcomes.

The modelling target is:

- `Churn = 1`: Customer left the provider
- `Churn = 0`: Customer remained with the provider

- Customers: 7,043
- Target churn rate: approximately 26.5%


## Business Impact

The business-impact simulation demonstrates how churn probabilities could support retention-budget allocation.

The simulation uses hypothetical assumptions for:

- Retention campaign cost
- Annual customer value
- Retention success rate

These figures are illustrative and should not be interpreted as an estimate of actual financial return.

## Installation

Clone the repository:

```bash
git clone https://github.com/Saptaparnineogi/Customer-Churn-Risk-Assessment-and-Retention-Analytics.git
cd Customer-Churn-Risk-Assessment-and-Retention-Analytics
````
````
conda create -n churn-prediction python=3.11
conda activate churn-prediction
````
````
pip install -r requirements.txt
````

## Model Training

```bash
python train.py

```
## Making Predictions

Generate churn predictions for unseen customer data using the trained model.

```bash
python predict.py \
    --input data/new_customers.csv \
    --output outputs/predictions.csv
```

The prediction pipeline automatically:

- loads the trained model and preprocessing pipeline
- applies the same feature engineering used during training
- generates churn probabilities
- applies the optimized decision threshold
- exports predictions to a CSV file

Example output:

| customerID | churn_probability | predicted_churn |
|------------|------------------:|----------------:|


>The exported CSV can be used directly for downstream business analysis, retention campaigns, or integration into reporting workflows.

## Testing

Run all unit tests:

```bash
pytest
```

The project includes automated tests for:

- preprocessing
- decision threshold optimization
- model persistence

Continuous integration is configured using GitHub Actions, ensuring that tests are executed automatically on every push.

## Limitations

- The dataset is relatively small and represents a simplified telecom use case.
- Historical campaign-response data and customer lifetime value were unavailable.
- The financial-impact simulation uses illustrative assumptions.
- Feature importance and SHAP values explain model behaviour but do not establish causal relationships.
- The model has not been validated on an external or more recent customer population.


## Future Improvements

- Probability calibration
- Customer lifetime value modelling
- Profit-based threshold optimisation
- Hyperparameter tuning with Optuna
- Model monitoring and drift detection
- Deployment through a Streamlit application or REST API


## Conclusion

This project demonstrates an end-to-end machine learning workflow that extends beyond predictive modelling to business decision support.

By combining robust feature engineering, model benchmarking, explainable AI, and business-oriented evaluation, it illustrates how churn prediction can be translated into actionable customer retention strategies.

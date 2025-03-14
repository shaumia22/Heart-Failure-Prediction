# Heart-Failure-Prediction
Team 4 Project

Data: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download
## Project Overview - What?
This project aims to predict heart disease using a Data Science approach. It utilizes the “Heart Failure Prediction” dataset to analyze 11 key demographic, clinical, and exercise-related features that are significant predictors of heart disease. By employing multiple classifier models after thorough preprocessing and hyperparameter tuning, to support early detection and intervention for individuals with cardiovascular disease or identifying those at high risk due to factors such as hypertension, diabetes, and hyperlipidemia, the best performing predictive model is selected post evaluation to ensure timely medical interventions, ultimately improving patient outcomes and reducing the global burden of CVDs..

## Business Value - Why?
Cardiovascular diseases (CVDs) are the leading cause of death globally, claiming approximately 17.9 million lives annually and accounting for 31% of global mortality. A significant proportion of these deaths, four out of five, are attributed to heart attacks and strokes, with a substantial number occurring prematurely in individuals under the age of 70. Heart failure is a common consequence of CVDs, underscoring the need for early detection and management which results in:
Improved patient outcomes by assisting in early detection and timely medical intervention resulting in personalized treatment and improved patient quality of life.
Cost optimization but reducing unnecessary tests and hospitalization and promoting preventive care potential lowering costs with late-stage treatments.
Enhanced data driven decision-making for healthcare providers and/or policy makers.
Deployment at scale where the model can be used across various cardiac care environments.

## Business Impact - How?

Using the Heart Failure Prediction dataset from Keggle, our goal is to deploy existing models, evaluate them, and choose the best performing model to identify predictors of heart disease by undergoing the following steps : 
* Analyzed Features
* Trained Model
* Delivered Insights

Models used for this project:
* Random Forest
* KNN
* XGBoost
* Linear Regression
* Decision Trees

Model Evaluation metrics:
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Log Loss
* RMSE

## Risks & Limitations - Mitigation Strategies
It is important to acknowledge potential risks, limitations and ways to mitigate them to ensure transparency and reliability of the model and data visualizations. Tables below summarizes the risks & limitations and mitigation plans considered for them:

1. Data-Related Risks

* Risk of Bias in Dataset Sources: The dataset combines data from multiple locations, each with different disease prevalence rates. This may affect model predictions.
 Mitigation: We validated prevalence (55%) to ensure a balanced mix and used class weighting to reduce bias.

* Risk of Missing Demographic Information: No details on ethnicity, diet, smoking, or family history.
  * Mitigation: Acknowledge these missing factors and suggest future work incorporating additional clinical data.

* Risk of Potential Label Noise: Some diagnosis labels may be inconsistent due to multiple sources.
  * Mitigation: Applied data cleaning techniques and validated predictions across models.
* Risk of Missing or Incorrect Cholesterol Values: Some cholesterol values are recorded as **zero**, which is medically unlikely and suggests missing data. Impact on Model:
  * Zero values could affect feature distributions and impact model predictions. 
  * Imputing arbitrary values (e.g., mean or median) could introduce bias without medical justification. 
  * Mitigation: We retain these zero values as-is rather than replacing them with estimated numbers. 

2. Model-Related Risks:
* Risk of Model Selection Impact on Results: The type of model used affects not only accuracy but also interpretability and fairness. 
   * Mitigation: We tested multiple models (**Random Forest, KNN, Linear Regression, Decision Trees, XGBoost**) to compare performance, interpretability, and bias. Feature importance analysis (SHAP values) and fairness evaluation across subgroups (e.g., age, sex) to ensure model reliability.  
* Risk of Overfitting and underfitting: The model may learn dataset-specific patterns instead of generalizable trends.
  * Mitigation: Used cross-validation and hyperparameter tuning to reduce overfitting and underfitting.

3. Deployment Risks:
* Risk of False Diagnoses (False Positives & False Negatives): Incorrect predictions could lead to unnecessary anxiety or missed treatment.
  * Mitigation: Implemented multiple models to ensure the best parameters and results will be selected from the advanced data analysis perspective. However, it is recommended to perform human oversight and clinical verification before making decisions based on model outputs.
4. Ethical Risks:
* Risk of Data Privacy: In real-world applications, compliance with regulations and patient consent is essential.
  * Mitigation: Stated that this dataset is anonymized but advised caution for real-world deployments.

Methods and Technologies

We are using the pandas framework, matplotlib and sklearn libraries to conduct the data exploration, feature engineering and the model selection/training.
For data exploration, we devised the min, max, mean values for each of the variables along with some data visualization tools such as box plot to catch any outliers and missing values.
We encoded the following categorical values using OneHotEncoder:
* Sex
* ChestPainType
* RestingECG
* ExerciseAngina
* ST_Slope
We created the following new features based on existing feature data to categorize the data further for better predictive performance:
* Cholesterol Groups
* Age Groups
We standardized the [will need to identify which features we standardize and how], after which we split the data into a training and testing set.
We tested the data across the following 5 models and conducted cross-validation to evaluate each model’s performance:
* Decision Tree
* k-Nearest Neighbors
* Random Forest
* Logistic Regression
* XGBoost
We analyzed each model’s performance and compared them against each other to determine the best model for this dataset.
Lastly, we ran the split test data on the selected model to establish reliable predictors of cardiovascular disease.

## Project Plan

| Category | Deliverable | ETA | Comments |
| ---------| ------------| ----| ---------|
| Documentation | Draft Project repo & readme | Mar 12 | Done |
| Documentation | Define Preliminary Project Plan | Mar 13 | Done |
| Data Clean-up | Load Data | Mar 13 | Done |
| Data Clean-up | Explore the Data | Mar 13 | Done |
| Data Clean-up |Handle Missing Values (if needed) | Mar 14 | Done |
| Feature Engineering | Encoding Categorical Values | Mar 17 | To Do |
| Feature Engineering | Creating new features (Define Age Groups, Categorize Cholesterol Levels) | Mar 17 | To Do |
| Feature Engineering | Feature scaling/standardizing | Mar 17 | To Do |
| Feature Engineering | Visualize the numerical feature distribution | Mar 17 | To Do |
| Feature Engineering | Split the data - test and train datasets |
| Feature Selection | Choose and fit a model | Mar 17 | To Do |
| Hyperparameter Tuning | GridSearchCV (Cross Validation) | Mar 17 | To Do |
| Hyperparameter Tuning | Identify the optimal hyperparameter combination | Mar 17 | To Do |
| Model Analysis | Evaluate the Model | Mar 17 | To Do |
| Model Selection | Visualize and analyze evaluation score to support model choice | Mar 18 | To Do |
| Model Selection | Explain the model performance | Mar 19? | To Do |
| Model Selection | Save the model - pkl files | Mar 20? | To Do |
| Load, deploy and test the model | Preset data or User entered data | Mar 20? | To Do |
| Update Repo | Create Pull requests and push changes to main repo | Mar 21 | To Do |
| Finalize Documentation | Upload recordings | Mar 21 | To Do |
| Finalize Documentation | Upload final ReadME | Mar 21 | To Do |


## Modelling

## Data Analysis

## Results


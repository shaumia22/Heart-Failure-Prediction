## Feature Importance for Heart Disease Classification

### Research Question
> **Which demographic, clinical, and exercise-related features are the most significant predictors of heart disease?**

### Approach:
We analyzed heart disease prediction using **logistic regression**, evaluating the importance of **demographic, clinical, and exercise-related features**. The model helps identify **key risk factors**.

### Notebook Location:
The main analysis for this research question is in:
📂 **[`notebooks/LogisticRegression_Model_KhatM.ipynb`](notebooks/LogisticRegression_Model_KhatM.ipynb)**

### Key Findings:
- **Age, Cholesterol, and ST_Slope were among the most significant clinical predictors.**
- **Exercise-induced angina and fasting blood sugar had a moderate impact.**
- **Sex had lower predictive power than expected.**

### Next Steps:
- Compare with other models (**Random Forest, XGBoost**) to confirm feature importance.
- Conduct fairness evaluation (e.g., does the model underpredict for females?).
- Deploy the model as a REST API for real-world testing.


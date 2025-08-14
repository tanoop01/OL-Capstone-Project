# Technical Report: Mental Health in the Tech Workplace

## 1. Executive Summary

This report details the analysis of the 2014 Open Sourcing Mental Illness (OSMI) Mental Health in Tech Survey. The primary objective was to uncover key factors influencing mental health treatment-seeking behavior among tech professionals. Through a combination of exploratory data analysis (EDA), predictive modeling, and unsupervised clustering, this project identifies critical workplace attributes and develops data-driven personas.

Key findings indicate that company-provided mental health benefits, clear and accessible leave policies, and a supportive culture are the strongest predictors of whether an employee will seek treatment. The analysis culminates in a set of actionable business recommendations designed to help tech companies foster a healthier and more supportive work environment. All models and insights are integrated into an interactive Streamlit dashboard for accessible exploration.

## 2. Problem Statement

The tech industry is often associated with high-stress environments, yet mental health remains a stigmatized topic. This creates a critical gap between the need for mental health support and the willingness or ability of employees to seek it.

The goal of this project is to address this gap by:
-   **Identifying Key Drivers**: Pinpointing the specific demographic and workplace factors that correlate with seeking mental health treatment.
-   **Building Predictive Tools**: Creating reliable machine learning models to predict treatment-seeking behavior and estimate employee age based on workplace context.
-   **Segmenting the Workforce**: Using clustering to identify distinct employee personas based on their perceptions and experiences with mental health in the workplace.
-   **Providing Actionable Insights**: Translating data-driven findings into concrete recommendations for businesses to improve employee well-being.

## 3. Dataset Description & Cleaning Summary

-   **Source**: The analysis is based on the 2014 OSMI Mental Health in Tech Survey, containing over 1,200 responses.
-   **Initial State**: The raw dataset included 27 columns with several data quality issues, including inconsistent values, outliers, and missing data.

**Data Cleaning Steps**:
-   **Dropped Columns**: `Timestamp`, `state`, and `comments` were removed as they were not relevant to the modeling objectives.
-   **Age Column**: Outlier values (ages below 0 and above 100) were removed to ensure data validity.
-   **Gender Column**: The `Gender` feature contained numerous free-text entries. These were standardized and consolidated into three distinct categories: 'Male', 'Female', and 'Other'.
-   **Missing Values**: Null values in `self_employed` and `work_interfere` were filled with 'No' and 'N/A' respectively, based on logical assumptions.
-   **Duplicates**: All duplicate rows were identified and removed to prevent data skew.
-   **Final Dataset**: The cleaned dataset was saved as `cleaned_data.csv` to serve as the foundation for all subsequent analysis and modeling.

## 4. Exploratory Data Analysis (EDA) with Visuals

A thorough EDA was conducted to uncover patterns and relationships within the data.

-   **Age Distribution**: The majority of respondents are between 25 and 40 years old, which is representative of the tech industry's workforce.
-   **Gender and Treatment**: While the survey had more male respondents, the proportion of individuals seeking treatment was relatively balanced across genders.
-   **Company Benefits**: A strong correlation was observed between companies offering mental health benefits and employees seeking treatment. Respondents from companies with benefits were significantly more likely to have sought help.
-   **Wellness Programs**: The presence of a company-sponsored wellness program also correlated positively with seeking treatment, suggesting that such initiatives help create a more supportive atmosphere.
-   **Leave Policy**: The ease of taking medical leave for a mental health condition was a critical factor. Employees who found it "Very easy" or "Somewhat easy" to take leave were far more likely to seek treatment than those who found it difficult or were unsure about the policy.
-   **Correlation Analysis**: A heatmap of all encoded features revealed that `treatment` has the highest positive correlation with `family_history`, `benefits`, and `care_options`. This reinforces that personal background and explicit company support are intertwined.

## 5. Classification Task: Predicting Treatment-Seeking Behavior

**Objective**: To build a model that accurately predicts whether an individual will seek mental health treatment.

-   **Preprocessing**: Categorical features were one-hot encoded, and the target variable `treatment` was label-encoded. The data was split into an 80% training set and a 20% testing set.
-   **Models Evaluated**:
    1.  Logistic Regression
    2.  Random Forest Classifier
    3.  XGBoost Classifier
    4.  Support Vector Machine (SVM)
-   **Results**: The models were evaluated based on Accuracy, ROC-AUC, and F1 Score.

| Model               | Accuracy | ROC-AUC | F1 Score |
| ------------------- | -------- | ------- | -------- |
| Logistic Regression | 0.839    | 0.898   | 0.841    |
| **Random Forest**   | **0.843**| 0.901   | **0.844**|
| XGBoost             | 0.827    | 0.892   | 0.828    |
| SVM                 | 0.819    | 0.881   | 0.813    |

-   **Discussion**: The **Random Forest Classifier** emerged as the best-performing model, achieving the highest accuracy and F1 score. Its feature importance analysis confirmed the EDA findings, highlighting `family_history`, `benefits`, and workplace culture indicators as the most influential predictors.

## 6. Regression Task: Predicting Age

**Objective**: To predict the age of a respondent based on their survey answers.

-   **Preprocessing**: The data was one-hot encoded, and the target variable `Age` was log-transformed to stabilize variance. A Random Forest model was first used for feature selection, identifying the top 90 most important features.
-   **Models Evaluated**:
    1.  Linear Regression
    2.  Random Forest Regressor
    3.  XGBoost Regressor
-   **Results**: After hyperparameter tuning, the models were evaluated using RMSE, MAE, and R² Score. The **Random Forest Regressor** again proved to be the most effective model.
    -   **RMSE**: 7.85
    -   **MAE**: 5.98
    -   **R² Score**: 0.35
-   **Discussion**: While the R² score indicates that the model explains a moderate portion of the variance, the feature importance plot was insightful. It showed that variables related to work (`no_employees`), health (`work_interfere`), and geography (`Country`) were the strongest predictors of age.

## 7. Clustering Analysis & Interpretation

**Objective**: To segment respondents into distinct personas using unsupervised learning.

-   **Methodology**: Key features related to workplace culture and attitudes were selected. After label encoding and scaling, K-Means clustering was applied. The optimal number of clusters was determined to be **3** using the Elbow Method and Silhouette Score analysis.
    -   **Silhouette Score**: A score of **0.196** was achieved with 3 clusters, indicating a reasonable separation.
-   **Persona Interpretation**:
    1.  **Cluster 0: Open Advocates**: This group is characterized by a high likelihood of having a family history of mental illness and actively seeking treatment. They tend to work for companies with strong mental health benefits and supportive policies.
    2.  **Cluster 1: Under-Supported Professionals**: This persona often works in environments where mental health benefits are unknown or unavailable. They are less likely to have sought treatment, possibly due to a lack of resources or a less open culture.
    3.  **Cluster 2: Silent Sufferers**: This group is the least likely to seek treatment. They often perceive negative consequences for discussing mental health at work and may work in companies with unclear or unsupportive leave policies.

## 8. Deployment Details

The entire analysis, including the predictive models and clustering results, was consolidated into an interactive web application using **Streamlit**.

-  It can be easily accessed and opened by using this url : [Streamlit App](https://ol-capstone-project-ziazg8daf6miapp226ea2sk.streamlit.app/)

## 9. Business Recommendations

Based on the analysis, the following recommendations are proposed for tech companies:

1.  **Invest in and Clearly Communicate Mental Health Benefits**: This is the most impactful factor. Ensure benefits are comprehensive and that employees know how to access them.
2.  **Create a Clear and Flexible Leave Policy**: An ambiguous or difficult leave policy is a major barrier. The process for taking mental health leave should be as straightforward as for physical health.
3.  **Promote an Open and Supportive Culture**: Encourage open conversations about mental health through wellness programs and leadership training. This reduces stigma and empowers employees to seek help.
4.  **Train Managers and Supervisors**: Equip leadership with the skills to recognize signs of distress and guide employees toward available resources compassionately and confidentially.

## 10. Challenges Faced & Future Scope

-   **Challenges**:
    -   **Outdated Data**: The dataset is from 2014. Attitudes and workplace policies have evolved since then.
    -   **Self-Reported Data**: The data is subject to response bias, where individuals may not answer truthfully due to stigma.
    -   **Class Imbalance**: While not severe, there was a slight imbalance in the `treatment` variable, which was addressed through model selection.
-   **Future Scope**:
    -   **Analyze Recent Data**: Replicating the analysis with a more current dataset would provide more relevant insights.
    -   **Analyze Text Data**: The 'comments' column could be analyzed using NLP techniques to extract qualitative insights.
    -   **Causal Inference**: Conduct studies to determine the causal impact of specific policy interventions on employee well-being.


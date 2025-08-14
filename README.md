# Mental Health in the Tech Workplace: An Analytical Dashboard

## Overview

This project analyzes the 2014 OSMI (Open Sourcing Mental Illness) Mental Health in Tech Survey data to uncover insights into the mental well-being of professionals in the tech industry. It features a comprehensive exploratory data analysis (EDA), predictive modeling for treatment-seeking behavior and age, and persona clustering to identify different employee attitudes towards mental health.

All these components are presented in an interactive web application built with Streamlit, providing a user-friendly interface to explore the data and model predictions.

## Problem Statement

The primary goal of this project is to create an interactive dashboard that allows users to:

-   **Visualize Key Trends**: Explore visualizations of demographics, workplace conditions, and attitudes towards mental health from the survey data.
-   **Predict Treatment Likelihood**: Use a machine learning model to predict whether an individual is likely to seek mental health treatment based on their background and workplace environment.
-   **Estimate Respondent Age**: Predict a respondent's age based on their survey answers.
-   **Discover Employee Personas**: Use clustering to identify and understand different employee personas, such as "Open Advocates," "Under-Supported Professionals," and "Silent Sufferers."

## Dataset

The project uses the **OSMI Mental Health in Tech Survey** dataset, which is publicly available on [Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey). This dataset contains over 1,200 responses from tech employees, covering topics like:

-   Demographics (Age, Gender, Country)
-   Workplace environment (company size, remote work, tech company status)
-   Attitudes towards mental health (benefits, wellness programs, anonymity, leave policies)
-   Personal and family history of mental illness

## Features of the Dashboard

The Streamlit application is organized into several pages:

1.  **Introduction**: Provides context on the importance of mental health in the tech industry and outlines the project's goals.
2.  **Exploratory Data Analysis (EDA)**: A comprehensive gallery of plots and charts visualizing the distributions and relationships within the data.
3.  **Treatment Prediction**: An interactive form where users can input features to get a prediction on the likelihood of seeking treatment, complete with model confidence.
4.  **Age Prediction**: An interactive form to predict a respondent's age based on their workplace and health-related features.
5.  **Persona Clustering**: A visualization of employee personas and a tool to classify a user into a persona based on their inputs.

## Technologies Used

-   **Language**: Python
-   **Web Framework**: Streamlit
-   **Data Manipulation**: Pandas, NumPy
-   **Data Visualization**: Matplotlib, Seaborn
-   **Machine Learning**: Scikit-learn, XGBoost

## Setup and Installation
The entire analysis, including the predictive models and clustering results, was consolidated into an interactive web application using Streamlit.

It can be easily accessed and opened by using this url : [Streamlit App](https://ol-capstone-project-ziazg8daf6miapp226ea2sk.streamlit.app/)

To run this project locally, follow these steps:

1.  **Clone the repository** (or ensure all project files are in the same directory).

2.  **Create and activate a virtual environment**:
    ```bash
    # Create a virtual environment
    python -m venv myenv

    # Activate on Windows
    .\myenv\Scripts\Activate.ps1

    # Activate on macOS/Linux
    source myenv/bin/activate
    ```

3.  **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application**:
    ```bash
    streamlit run streamlit_app.py
    ```

## File Descriptions

-   `streamlit_app.py`: The main script for the interactive Streamlit web dashboard.
-   `EDA.ipynb`: Jupyter Notebook containing the code for data cleaning, preprocessing, and detailed exploratory data analysis.
-   `classification_model.ipynb`: Jupyter Notebook for building and evaluating various models (Logistic Regression, Random Forest, XGBoost, SVM) to predict treatment-seeking behavior.
-   `regression_model.ipynb`: Jupyter Notebook for building and evaluating models to predict the age of respondents.
-   `clustering.ipynb`: Jupyter Notebook that uses K-Means clustering to group employees into different personas based on their survey responses.
-   `survey.csv`: The original, raw dataset from the OSMI survey.
-   `cleaned_data.csv`: The processed dataset after cleaning and feature engineering, used by the models and the dashboard.
-   `requirements.txt`: A list of all Python dependencies required to run the project.
-   `README.md`: This file, providing an overview and instructions for the project.


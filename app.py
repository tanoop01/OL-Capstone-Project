import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load and cache the data
@st.cache_data
def load_data():
    df = pd.read_csv('survey.csv')
    return df

# Preprocessing function
def preprocess_data(df):
    # Basic cleaning
    df = df.drop(columns=['Timestamp', 'state', 'comments'], errors='ignore')
    df.drop(df[df['Age'] < 0].index, inplace = True) 
    df.drop(df[df['Age'] > 100].index, inplace = True)
    df['self_employed'].fillna('No', inplace=True)
    df['work_interfere'].fillna('N/A', inplace=True)

    # Clean 'Gender'
    df['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                         'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                         'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'Male', inplace=True)
    df['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                         'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                         'woman',], 'Female', inplace=True)
    df['Gender'].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                         'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                         'Agender', 'A little about you', 'Nah', 'All',
                         'ostensibly male, unsure what that really means',
                         'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                         'Guy (-ish) ^_^', 'Trans woman',], 'Other', inplace=True)
    
    return df

df_raw = load_data()
df_processed = preprocess_data(df_raw.copy())

# st.title("Mental Health in Tech-Dashboard")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction","About Data", "Exploratory Data Analysis", "Treatment Prediction", "Age Prediction", "Persona Clustering"])

if page == "Introduction":
    st.title("Mental Health in Tech-Dashboard")
    st.header("Introduction")
    st.write("""
    This application provides insights into mental health in the tech industry based on the OSMI Mental Health in Tech Survey.
    You can explore the data, predict treatment-seeking behavior, estimate age, and discover different mental health personas within the tech workforce.
    Use the sidebar to navigate through the different sections of the app.
    """)
elif page == "About Data" :

    st.write("### Data Summary")

    # Initialize state
    if "show_all" not in st.session_state:
        st.session_state.show_all = False

# Function to toggle state
    def toggle_rows():
        st.session_state.show_all = not st.session_state.show_all

# Display based on state
    if st.session_state.show_all:
        st.write(df_processed)
    else:
        st.write(df_processed.head())

# Button with dynamic label (now below the table)
    st.button(
        "Show all rows" if not st.session_state.show_all else "Show less",
        on_click=toggle_rows
    )

elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    # --- Age Distribution ---
    st.subheader("Age Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df_processed['Age'], kde=True, ax=ax1)
    st.pyplot(fig1)
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf1.getvalue(),
        file_name="age_distribution.png",
        mime="image/png",
        key='age_dist_download'
    )

    # --- Treatment Seeking by Gender ---
    st.subheader("Treatment Seeking by Gender")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_processed, x='treatment', hue='Gender', ax=ax2)
    st.pyplot(fig2)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf2.getvalue(),
        file_name="treatment_by_gender.png",
        mime="image/png",
        key='gender_treat_download'
    )

    # --- Family History ---
    st.subheader("Family History of Mental Illness")
    fig3, ax3 = plt.subplots()
    df_processed['family_history'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax3)
    plt.ylabel('')
    st.pyplot(fig3)
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf3.getvalue(),
        file_name="family_history.png",
        mime="image/png",
        key='fam_hist_download'
    )


# elif page == "Exploratory Data Analysis":
#     st.header("Exploratory Data Analysis")
    
#     # --- Age Distribution ---
#     st.subheader("Age Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(df_processed['Age'], kde=True, ax=ax)
#     st.pyplot(fig)

#     # Save button for Age Distribution
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png")
#     buf.seek(0)
#     st.download_button(
#         label="💾 Save Age Distribution",
#         data=buf,
#         file_name="age_distribution.png",
#         mime="image/png"
#     )

#     # --- Treatment Seeking by Gender ---
#     st.subheader("Treatment Seeking by Gender")
#     fig, ax = plt.subplots()
#     sns.countplot(data=df_processed, x='treatment', hue='Gender', ax=ax)
#     st.pyplot(fig)

#     buf = io.BytesIO()
#     fig.savefig(buf, format="png")
#     buf.seek(0)
#     st.download_button(
#         label="💾 Save Treatment by Gender",
#         data=buf,
#         file_name="treatment_by_gender.png",
#         mime="image/png"
#     )

#     # --- Family History ---
#     st.subheader("Family History of Mental Illness")
#     fig, ax = plt.subplots()
#     df_processed['family_history'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
#     plt.ylabel('')
#     st.pyplot(fig)

#     buf = io.BytesIO()
#     fig.savefig(buf, format="png")
#     buf.seek(0)
#     st.download_button(
#         label="💾 Save Family History Chart",
#         data=buf,
#         file_name="family_history.png",
#         mime="image/png"
#     )

elif page == "Treatment Prediction":
    st.header("Predict if a Person Will Seek Treatment")

    # Features for classification
    clf_features = df_processed.drop(columns=['treatment', 'Age']).columns
    
    # One-hot encode for the model
    df_encoded = pd.get_dummies(df_processed, drop_first=True)
    
    X = df_encoded.drop('treatment_Yes', axis=1)
    y = df_encoded['treatment_Yes']
    
    # Train model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    st.sidebar.header("Your Information")
    
    input_data = {}
    for col in clf_features:
        if df_processed[col].dtype == 'object':
            options = df_processed[col].unique()
            input_data[col] = st.sidebar.selectbox(f"Select {col}", options)
        else:
            input_data[col] = st.sidebar.number_input(f"Enter {col}", value=float(df_processed[col].mean()))

    if st.sidebar.button("Predict Treatment"):
        input_df = pd.DataFrame([input_data])
        
        # Create a full df with all possible columns from training
        input_encoded = pd.get_dummies(input_df)
        input_aligned = input_encoded.reindex(columns=X.columns, fill_value=0)
        
        prediction = clf.predict(input_aligned)[0]
        prediction_proba = clf.predict_proba(input_aligned)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.write("The model predicts this person is **likely** to seek treatment.")
            st.write(f"Confidence: {prediction_proba[1]:.2f}")
        else:
            st.write("The model predicts this person is **not likely** to seek treatment.")
            st.write(f"Confidence: {prediction_proba[0]:.2f}")

elif page == "Age Prediction":
    st.header("Predict Respondent's Age")

    # Define features for regression
    reg_features = df_processed.drop(columns=['Age']).columns

    # One-hot encode for the model
    df_encoded = pd.get_dummies(df_processed, drop_first=True)
    
    X = df_encoded.drop('Age', axis=1)
    y = df_encoded['Age']

    # Train model
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X, y)

    st.sidebar.header("Workplace & Health Features")
    input_data = {}
    for col in reg_features:
        if df_processed[col].dtype == 'object':
            options = df_processed[col].unique()
            # Use a unique key for each widget
            input_data[col] = st.sidebar.selectbox(f"Select {col}", options, key=f"age_{col}")
        else:
            # This case is not expected for this dataset's features, but good practice
            input_data[col] = st.sidebar.number_input(f"Enter {col}", value=float(df_processed[col].mean()), key=f"age_{col}")

    if st.sidebar.button("Predict Age"):
        input_df = pd.DataFrame([input_data])
        
        # Create a full df with all possible columns from training
        input_encoded = pd.get_dummies(input_df)
        input_aligned = input_encoded.reindex(columns=X.columns, fill_value=0)
        
        predicted_age = reg.predict(input_aligned)
        
        st.subheader("Prediction Result")
        st.write(f"The predicted age for the given profile is: **{int(predicted_age[0])}**")


elif page == "Persona Clustering":
    st.header("Discover Your Mental Health Persona")
    
    features = ['family_history', 'treatment', 'mental_health_consequence', 'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave']
    X_cluster = df_processed[features].copy()

    # Label encode all features
    le_dict = {}
    for col in X_cluster.columns:
        le = LabelEncoder()
        X_cluster[col] = le.fit_transform(X_cluster[col])
        le_dict[col] = le

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Train KMeans3
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)
    df_processed['cluster'] = clusters

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_processed['pca1'] = X_pca[:, 0]
    df_processed['pca2'] = X_pca[:, 1]

    cluster_names = {
        0: "Open Advocates",
        1: "Under-Supported Professionals",
        2: "Silent Sufferers"
    }
    df_processed['persona'] = df_processed['cluster'].map(cluster_names)

    st.sidebar.header("Your Attitudes & Workplace")
    input_data = {}
    for col in features:
        options = df_raw[col].unique()
        input_data[col] = st.sidebar.selectbox(f"Select {col}", options, key=col)

    if st.sidebar.button("Find My Persona"):
        input_df = pd.DataFrame([input_data])
        
        for col in input_df.columns:
            input_df[col] = le_dict[col].transform(input_df[col])
            
        input_scaled = scaler.transform(input_df)
        persona_cluster = kmeans.predict(input_scaled)[0]
        persona_name = cluster_names[persona_cluster]

        st.subheader(f"Your Persona: {persona_name}")
        
        if persona_name == "Open Advocates":
            st.write("You are likely open about your mental health, have sought treatment, and work in a supportive environment.")
        elif persona_name == "Under-Supported Professionals":
            st.write("You may work for a company that offers benefits but lacks strong mental health support systems like wellness programs or easy leave.")
        elif persona_name == "Silent Sufferers":
            st.write("You might be hesitant to discuss mental health, possibly due to concerns about negative consequences, and may not be aware of or have access to company resources.")

    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df_processed, x='pca1', y='pca2', hue='persona', palette='viridis', ax=ax)
    plt.title('Mental Health Personas in Tech')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="persona_clusters.png",
        mime="image/png",
        key='persona_cluster_download'
    )

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Loading original dataset
raw_data = pd.read_csv("survey.csv")
# Load and cache the data
@st.cache_data
def load_data():
    # Using the cleaned data as it's already processed
    df = pd.read_csv("cleaned_data.csv")
    return df

df_processed = load_data()


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Problem Statement", "About Data", "Exploratory Data Analysis", "Treatment Prediction", "Age Prediction", "Persona Clustering"])

if page == "Introduction" :
    st.title("Mental Health in Tech Workforce")
    st.header("Why It Matters")
    st.write("""
Mental wellbeing is not a solo issue—it's a business and societal necessity. The World Health Organization indicates that depression and anxiety alone cause the global economy a loss of more than USD 1 trillion every year in lost productivity. Productivity losses in India's tech industry due to employee wellbeing issues stand at a whopping ₹83 lakh crore each year""")
             
    st.header("The Specialized Stressors of Tech Spaces")
    st.write("""Tech environments have several specialized challenges that complicate mental health issues:
             
High-pressure, Always-on Culture
Short deadlines, innovation cycles, and a sense of perpetual connectivity push workers into burnout and chronic stress.
             
Techno-stress (Technostress)
Quick technological change, absence of standardization, and inadequate training lead to an effect called technostress, which has unambiguous psychological and performance-driven effects—such as insomnia, irritability, and decreased job satisfaction.

Burnout & Exhaustion
Global polls like the Burnout Index point out that 62% of technology employees are emotionally and physically drained, with almost half actively considering leaving. Likewise, 57% experience burnout, with high levels of stress and poor coping mechanisms being reported throughout the industry.

Job Insecurity & Layoffs
Technology's boom-and-bust lay-offs, particularly post-pandemic, keep everyone on edge. According to one survey, 52% of technology employees suffer from depression or anxiety, with many struggling with unclear futures. A popular Reddit thread highlighted extreme cases—such as experienced developers coerced into 20 hours of uncompensated overtime, resulting in extreme stress and health problems.

Surveillance & "Bossware"
Growth of AI-powered monitoring software leads to a culture of over-surveillance. Almost 45% of employees reported heightened stress, with 29–34% blaming increased pressure from such software.

Content Moderator Trauma
Employees who are exposed to toxic content (e.g., on Meta, TikTok, Google) experience deep psychological burdens. In turn, they've mobilized around the world to call for stronger protections and support
The Verge """)


    st.header("The Cost of Poor Psyche: More Than Dollars")
    st.write("""
A poor psychosocial safety climate (PSC)—that is, an organization's collective perception about psychological well-being—has quantifiable costs. Low PSC has the potential to triple the rate of depressive symptoms over the course of a year, and interventions can have a significant impact on reducing burnout and absenteeism.

In the UK, poor mental health at work cost employers £56 billion in 2020–2021—yet spending £1 on well-being apparently brings in an average of £5.30. In Australia, removing low PSC could decrease sickness absence by 14% and presenteeism by 72%. """)


    st.header("Creating a Supportive Tech Workplace: Best Practices")
    st.write("""1. Foster Awareness & Preventive Culture

Utilize Employee Assistance Programs (EAPs) providing counseling and support. Train managers and teams to recognize distress and respond with empathy.

2. Foster Ethical & Adaptive Leadership

Leaders must set the example—shutting off, honoring weekends, and speaking openly about stress.
Policies must prioritize realistic expectations, work hour caps, and anti-harassment policies.

3. Create People-Centric Work Policies

Implement initiatives such as "no-screen Fridays" or "Me Days" to mitigate burnout and validate employee well-being.
Providing four-day workweeks in a considered rollout (not condensing hours) has been shown to be effective in reducing stress

4. Promote Physical Activity & Boundary Setting

Physical movement is essential: WHO advises a minimum of 150 minutes moderate activity a week. Tech leaders need to bring this into infrastructure, not address it as an afterthought.
Clear boundaries between work and personal time assist with eradicating the "always-on" culture.
             
5. Promote Technostress Through Innovation

Facilitate training, supportive leadership, and constructive tech culture to alleviate technostress and enhance job performance.
Encourage individual innovation to buffer the negative effects of technostress and enhance organizational commitment.
             
6. Invest Organizationally in Employee Well-Being

Support organizational commitment and ethical climates to minimize tech-induced fears such as nomophobia. 
In engineering in particular, aspects such as collaborative culture, appreciation, and organizational support are all crucial for sustaining well-being.    """)


    st.header("Conclusion")
    st.write("""In the high-speed tech world, neglecting mental health isn't only damaging—it's not sustainable. High-stress situations, technostress, burnout, and monitoring impose severe costs. But here's the bright spot: employers have power. By integrating well-being into leadership, organization, culture, and design, tech companies can create settings in which innovation bursts forth without compromising mental health.
             """)


elif page == "Problem Statement":
    st.title("Problem Statement")
    st.header("Create an interactive web app with:")
    st.write(""" > EDA visualizations
             
> Input form to predict whether a person will seek treatment

> Display of predicted age based on mental health and workplace features

> Cluster visualizer with description of user persona

> Include model confidence, data summary, and recommendations """)
    
    st.header("About this dashboard : ")
    
    st.write("""
             
   > This application provides insights into mental health in the tech industry based on the [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey) .
    You can explore the data, predict treatment-seeking behavior, estimate age, and discover different mental health personas within the tech workforce.
    Use the sidebar to navigate through the different sections of the app.
    """)


elif page == "About Data" :
    st.title("About the Data")
    st.write("This dataset is from the 2014 OSMI Mental Health in Tech Survey. It includes responses from tech professionals about their mental health attitudes and workplace conditions.")

    st.write("Dataset Link : [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)")
    
    st.write("### Original Data Summary")
    
    # Initialize state for raw data
    if "show_all_raw" not in st.session_state:
        st.session_state.show_all_raw = False

    # Function to toggle state for raw data
    def toggle_rows_raw():
        st.session_state.show_all_raw = not st.session_state.show_all_raw

    # Display based on state
    if st.session_state.show_all_raw:
        st.write(raw_data)
    else:
        st.write(raw_data.head())

    # Button with dynamic label and unique key
    st.button(
        "Show all rows" if not st.session_state.show_all_raw else "Show less",
        on_click=toggle_rows_raw,
        key='toggle_raw_data'
    )

    st.write("### Cleaned Data Summary")

    # Initialize state for cleaned data
    if "show_all_cleaned" not in st.session_state:
        st.session_state.show_all_cleaned = False

    # Function to toggle state for cleaned data
    def toggle_rows_cleaned():
        st.session_state.show_all_cleaned = not st.session_state.show_all_cleaned

    # Display based on state
    if st.session_state.show_all_cleaned:
        st.write(df_processed)
    else:
        st.write(df_processed.head())

    # Button with dynamic label and unique key
    st.button(
        "Show all rows" if not st.session_state.show_all_cleaned else "Show less",
        on_click=toggle_rows_cleaned,
        key='toggle_cleaned_data'
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

    # --- Gender Distribution vs Treatment ---
    st.subheader("Gender Distribution vs. Treatment")
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', hue='treatment',data=df_processed, order=df_processed['Gender'].value_counts().index, ax=ax)
    ax.set_title('Gender Distribution vs. Treatment')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="gender_vs_treatment.png",
        mime="image/png",
        key='gender_vs_treatment_download'
    )

    # --- Treatment Distribution ---
    st.subheader("Overall Treatment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='treatment', data=df_processed, ax=ax)
    ax.set_title('Distribution of Seeking Treatment')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="treatment_distribution.png",
        mime="image/png",
        key='treatment_dist_download'
    )

    # --- Self Employment Distribution ---
    st.subheader("Self-Employment vs. Treatment")
    fig, ax = plt.subplots()
    sns.countplot(x='self_employed', hue='treatment',data=df_processed, order=df_processed['self_employed'].value_counts().index, ax=ax)
    ax.set_title('Self Employment Distribution')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="self_employment_vs_treatment.png",
        mime="image/png",
        key='self_employment_download'
    )

    # --- Family History Distribution ---
    st.subheader("Family History vs. Treatment")
    fig, ax = plt.subplots()
    sns.countplot(x='family_history', hue='treatment', data=df_processed, order=df_processed['family_history'].value_counts().index, ax=ax)
    ax.set_title('Family History Distribution')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="family_history_vs_treatment.png",
        mime="image/png",
        key='family_history_countplot_download'
    )

    # --- Work Interference Distribution ---
    st.subheader("Work Interference vs. Treatment")
    fig, ax = plt.subplots()
    sns.countplot(x='work_interfere', hue='treatment', data=df_processed, order=df_processed['work_interfere'].value_counts().index, ax=ax)
    ax.set_title('Work Interference Distribution')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="work_interference_vs_treatment.png",
        mime="image/png",
        key='work_interference_download'
    )

    # --- Number of Employees Distribution ---
    st.subheader("Number of Employees vs. Treatment")
    fig, ax = plt.subplots()
    sns.countplot(x='no_employees', hue='treatment',data=df_processed, order=df_processed['no_employees'].value_counts().index, ax=ax)
    ax.set_title('Number of Employees Distribution')
    plt.xticks(rotation=90)
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="employees_vs_treatment.png",
        mime="image/png",
        key='employees_download'
    )

    # --- Remote Work Distribution ---
    st.subheader("Remote Work vs. Treatment")
    fig, ax = plt.subplots()
    sns.countplot(x='remote_work', hue='treatment',data=df_processed, order=df_processed['remote_work'].value_counts().index, ax=ax)
    ax.set_title('Remote Work Distribution')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="remote_work_vs_treatment.png",
        mime="image/png",
        key='remote_work_download'
    )

    # --- Tech Company Distribution ---
    st.subheader("Work in a Tech Company vs. Treatment")
    fig, ax = plt.subplots()
    sns.countplot(x='tech_company', hue='treatment',data=df_processed, order=df_processed['tech_company'].value_counts().index, ax=ax)
    ax.set_title('Tech Company Distribution')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="tech_company_vs_treatment.png",
        mime="image/png",
        key='tech_company_download'
    )

    # --- Company Benefits vs. Treatment ---
    st.subheader("Company Benefits vs. Treatment")
    fig, ax = plt.subplots()
    sns.countplot(x='benefits', hue='treatment', data=df_processed, ax=ax)
    ax.set_title('Effect of Company Benefits on Seeking Treatment')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="benefits_vs_treatment.png",
        mime="image/png",
        key='benefits_download'
    )

    # --- Wellness Program vs. Treatment ---
    st.subheader("Wellness Program vs. Treatment")
    fig, ax = plt.subplots()
    sns.countplot(x='wellness_program', hue='treatment', data=df_processed, ax=ax)
    ax.set_title('Effect of Wellness Programs on Seeking Treatment')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="wellness_vs_treatment.png",
        mime="image/png",
        key='wellness_download'
    )

    # --- Ease of Leave vs. Treatment ---
    st.subheader("Ease of Taking Medical Leave vs. Treatment")
    fig, ax = plt.subplots()
    sns.countplot(x='leave', hue='treatment', data=df_processed, order=['Very easy', 'Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult'], ax=ax)
    ax.set_title('Effect of Leave Policy on Seeking Treatment')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="leave_vs_treatment.png",
        mime="image/png",
        key='leave_download'
    )

    # --- Correlation Heatmap ---
    st.subheader("Feature Correlation Heatmap")
    df_encoded = df_processed.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    corr_matrix = df_encoded.corr()
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix of All Features')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Download",
        data=buf.getvalue(),
        file_name="correlation_heatmap.png",
        mime="image/png",
        key='heatmap_download'
    )

    st.subheader("Top Correlations with Treatment")
    corr_treatment = corr_matrix['treatment'].sort_values(ascending=False)
    st.write("Top 5 positive correlations with seeking treatment:")
    st.write(corr_treatment.head(5))
    st.write("Top 5 negative correlations with seeking treatment:")
    st.write(corr_treatment.tail(5))

elif page == "Treatment Prediction":
    st.header("Predict if a Person Will Seek Treatment")

    # Initialize session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None

    # Prepare data
    X = df_processed.drop('treatment', axis=1)
    y = df_processed['treatment']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Model Training
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])
    
    clf.fit(X, y_encoded)

    st.sidebar.header("Your Information")
    
    input_data = {}
    for col in X.columns:
        if col in numerical_features:
            input_data[col] = st.sidebar.number_input(f"Enter {col}", value=float(df_processed[col].mean()))
        else:
            options = df_processed[col].unique()
            input_data[col] = st.sidebar.selectbox(f"Select {col}", options)

    if st.sidebar.button("Predict Treatment"):
        input_df = pd.DataFrame([input_data])
        
        prediction = clf.predict(input_df)[0]
        prediction_proba = clf.predict_proba(input_df)[0]
        
        st.session_state.prediction_result = {
            "prediction": prediction,
            "probability": prediction_proba
        }

    # Display placeholder or result
    if st.session_state.prediction_result is None:
        st.markdown("<h1 style='text-align: center; font-size: 200px; color: #ccc;'>?</h1>", unsafe_allow_html=True)
    else:
        result = st.session_state.prediction_result
        prediction = result['prediction']
        prediction_proba = result['probability']
        
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.write("The model predicts this person is **likely** to seek treatment. 😔")
            st.write(f"Confidence: {prediction_proba[1]:.2f}")
            st.subheader("Recommendations")
            st.write("""
            - **Encourage Open Dialogue**: Foster a work environment where employees feel safe to discuss mental health.
            - **Provide Resources**: Ensure easy access to mental health benefits, care options, and wellness programs.
            - **Flexible Leave**: Offer flexible leave policies for mental health needs.
            """)
        else:
            st.write("The model predicts this person is **not likely** to seek treatment. 😊")
            st.write(f"Confidence: {prediction_proba[0]:.2f}")
            st.subheader("Recommendations")
            st.write("""
            - **Proactive Support**: Even if an employee is not predicted to seek treatment, continue to promote mental health awareness.
            - **Anonymous Support**: Provide anonymous channels for seeking help.
            - **Regular Check-ins**: Encourage managers to have regular, supportive check-ins with their team members.
            """)

elif page == "Age Prediction":
    st.header("Predict Respondent's Age")

    # Initialize session state
    if 'predicted_age' not in st.session_state:
        st.session_state.predicted_age = None

    # Prepare data
    X = df_processed.drop('Age', axis=1)
    y = df_processed['Age']

    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Train model
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_encoded, y)

    st.sidebar.header("Workplace & Health Features")
    input_data = {}
    for col in X.columns:
        options = df_processed[col].unique()
        input_data[col] = st.sidebar.selectbox(f"Select {col}", options, key=f"age_{col}")

    if st.sidebar.button("Predict Age"):
        input_df = pd.DataFrame([input_data])
        
        # Create a full df with all possible columns from training
        input_encoded = pd.get_dummies(input_df)
        input_aligned = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)
        
        predicted_age_value = reg.predict(input_aligned)
        st.session_state.predicted_age = int(predicted_age_value[0])

    # Display placeholder or result
    if st.session_state.predicted_age is None:
        st.markdown("<h1 style='text-align: center; font-size: 200px; color: #ccc;'>?</h1>", unsafe_allow_html=True)
    else:
        st.subheader("Prediction Result")
        age = st.session_state.predicted_age
        emoji = ""
        if 1 <= age <= 35:
            emoji = "🧒🏻"
        elif 36 <= age <= 55:
            emoji = "👨🏻"
        elif 56 <= age <= 99:
            emoji = "🧑🏻‍🦳"
        st.write(f"The predicted age for the given profile is: **{age}** {emoji}")


elif page == "Persona Clustering":
    st.header("Discover Your Mental Health Persona")

    # Initialize session state
    if 'persona_name' not in st.session_state:
        st.session_state.persona_name = None
    
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

    # Train KMeans
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
        options = df_processed[col].unique()
        input_data[col] = st.sidebar.selectbox(f"Select {col}", options, key=col)

    if st.sidebar.button("Find My Persona"):
        input_df = pd.DataFrame([input_data])
        
        for col in input_df.columns:
            input_df[col] = le_dict[col].transform(input_df[col])
            
        input_scaled = scaler.transform(input_df)
        persona_cluster = kmeans.predict(input_scaled)[0]
        st.session_state.persona_name = cluster_names[persona_cluster]

    # Display placeholder or result
    if st.session_state.persona_name is None:
        st.markdown("<h1 style='text-align: center; font-size: 200px; color: #ccc;'>?</h1>", unsafe_allow_html=True)
    else:
        persona_name = st.session_state.persona_name
        emoji = ""
        description = ""
        if persona_name == "Open Advocates":
            emoji = "🧑🏻‍⚖"
            description = "You are likely open about your mental health, have sought treatment, and work in a supportive environment."
        elif persona_name == "Under-Supported Professionals":
            emoji = "👨🏻‍🏫"
            description = "You may work for a company that offers benefits but lacks strong mental health support systems like wellness programs or easy leave."
        elif persona_name == "Silent Sufferers":
            emoji = "🤦🏻‍♂"
            description = "You might be hesitant to discuss mental health, possibly due to concerns about negative consequences, and may not be aware of or have access to company resources."
        
        st.subheader(f"Your Persona: {persona_name} {emoji}")
        st.write(description)

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

st.sidebar.markdown("---")
st.sidebar.markdown("GitHub Repository : [OL-Capstone-Project](https://github.com/tanoop01/OL-Capstone-Project/tree/main)")




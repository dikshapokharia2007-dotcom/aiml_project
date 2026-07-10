
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


# PAGE CONFIG

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# LOAD MODEL + SCALER

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    return model

try:
    model = load_artifacts()
    LOAD_ERROR = None
except Exception as e:
    model = None
    LOAD_ERROR = str(e)

# LOAD TEST SET + RAW DATA FOR DASHBOARD 

@st.cache_resource
def load_test_data():
    # Rebuilt to exactly match the split used in model_training.ipynb
    # (train_test_split(test_size=0.2, random_state=42, stratify=y) on
    # raw, unscaled data straight from data/hr.csv). model.pkl was fit
    # directly on this — it never used scaler.pkl.
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("data/hr.csv")
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

@st.cache_data
def load_hr_data():
    df = pd.read_csv("data/hr.csv")
    return df

def decode_dummies(df):
    """Reconstruct human-readable categorical columns from the one-hot
    dummy columns saved in data/hr.csv, using the same drop_first baseline
    logic that pd.get_dummies used originally."""
    d = df.copy()

    d["Department"] = np.select(
        [d.get("Dept_Research & Development", 0) == 1, d.get("Dept_Sales", 0) == 1],
        ["Research & Development", "Sales"],
        default="Human Resources"
    )

    d["BusinessTravel"] = np.select(
        [d.get("Travel_Travel_Frequently", 0) == 1, d.get("Travel_Travel_Rarely", 0) == 1],
        ["Travel_Frequently", "Travel_Rarely"],
        default="Non-Travel"
    )

    d["EducationField"] = np.select(
        [
            d.get("EduField_Life Sciences", 0) == 1, d.get("EduField_Marketing", 0) == 1,
            d.get("EduField_Medical", 0) == 1, d.get("EduField_Other", 0) == 1,
            d.get("EduField_Technical Degree", 0) == 1,
        ],
        ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
        default="Human Resources"
    )

    d["Gender"] = np.where(d.get("Gender_Male", 0) == 1, "Male", "Female")

    role_cols = [
        ("JobRole_Human Resources", "Human Resources"),
        ("JobRole_Laboratory Technician", "Laboratory Technician"),
        ("JobRole_Manager", "Manager"),
        ("JobRole_Manufacturing Director", "Manufacturing Director"),
        ("JobRole_Research Director", "Research Director"),
        ("JobRole_Research Scientist", "Research Scientist"),
        ("JobRole_Sales Executive", "Sales Executive"),
        ("JobRole_Sales Representative", "Sales Representative"),
    ]
    conditions = [d.get(col, 0) == 1 for col, _ in role_cols]
    choices = [label for _, label in role_cols]
    d["JobRole"] = np.select(conditions, choices, default="Healthcare Representative")

    d["MaritalStatus"] = np.select(
        [d.get("Marital_Married", 0) == 1, d.get("Marital_Single", 0) == 1],
        ["Married", "Single"],
        default="Divorced"
    )

    d["OverTime"] = np.where(d.get("OverTime_Yes", 0) == 1, "Yes", "No")
    d["AttritionLabel"] = np.where(d["Attrition"] == 1, "Yes", "No")

    return d


# EXACT FEATURE ORDER THE MODEL WAS TRAINED ON

FEATURE_ORDER = [
    'Age', 'DailyRate', 'HomeDist', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLvl', 'JobSatisfaction', 'MonthlyInc',
    'MonthlyRate', 'NumCompaniesWorked', 'SalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'Work_Life_Bal', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'Travel_Travel_Frequently', 'Travel_Travel_Rarely',
    'Dept_Research & Development', 'Dept_Sales', 'EduField_Life Sciences',
    'EduField_Marketing', 'EduField_Medical', 'EduField_Other',
    'EduField_Technical Degree', 'Gender_Male', 'JobRole_Human Resources',
    'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive',
    'JobRole_Sales Representative', 'Marital_Married', 'Marital_Single',
    'OverTime_Yes'
]


# PASTEL / ANIMATED CSS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #fdf2f8 0%, #f0f9ff 25%, #f5f3ff 50%, #fef3f2 75%, #f0fdf4 100%);
    background-size: 300% 300%;
    animation: gradientShift 18s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Hero title */
.hero-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f472b6, #a78bfa, #60a5fa, #34d399);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shine 6s linear infinite;
    margin-bottom: 0;
    padding-top: 0.5rem;
}
@keyframes shine {
    to { background-position: 300% center; }
}

.hero-subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 0.2rem;
    margin-bottom: 1.8rem;
    animation: fadeInUp 1s ease;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Card containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 20px !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.55);
    padding: 8px;
    border-radius: 16px;
    backdrop-filter: blur(6px);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 10px 18px;
    font-weight: 500;
    color: #6b7280;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #fbcfe8, #ddd6fe) !important;
    color: #4c1d95 !important;
    box-shadow: 0 4px 12px rgba(167,139,250,0.35);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #f9a8d4, #c4b5fd);
    color: #3b0764;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 2.2rem;
    font-weight: 600;
    font-size: 1.05rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 14px rgba(196,181,253,0.5);
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 8px 22px rgba(244,114,182,0.45);
    color: #3b0764;
}
.stButton > button:active {
    transform: translateY(0px) scale(0.98);
}

/* Sliders */
.stSlider [data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #f9a8d4, #a5b4fc) !important;
}

/* Result cards */
.result-card {
    border-radius: 24px;
    padding: 2.2rem 2rem;
    text-align: center;
    animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}
@keyframes popIn {
    0% { opacity: 0; transform: scale(0.85); }
    100% { opacity: 1; transform: scale(1); }
}
.stay-card {
    background: linear-gradient(135deg, #d1fae5, #ecfdf5);
    border: 1px solid #6ee7b7;
}
.leave-card {
    background: linear-gradient(135deg, #fee2e2, #fff1f2);
    border: 1px solid #fca5a5;
}
.result-emoji {
    font-size: 3.4rem;
    display: block;
    margin-bottom: 0.4rem;
    animation: bounce 1.6s ease infinite;
}
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
.result-heading {
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-sub {
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}

/* Probability bar */
.prob-bar-bg {
    width: 100%;
    height: 22px;
    border-radius: 999px;
    background: rgba(255,255,255,0.6);
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    animation: growBar 1.2s ease-out forwards;
}
@keyframes growBar {
    from { width: 0%; }
}

/* Section headers */
.section-header {
    font-weight: 600;
    color: #7c3aed;
    font-size: 1.05rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Force readable text everywhere, regardless of viewer's system dark/light mode */
label, .stMarkdown, .stMarkdown p, .stSelectbox label, .stSlider label,
.stRadio label, .stTextInput label, [data-testid="stWidgetLabel"] p {
    color: #374151 !important;
}
.stTabs [data-baseweb="tab"] p {
    color: #6b7280 !important;
}
.stTabs [aria-selected="true"] p {
    color: #4c1d95 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #374151 !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    color: #f9fafb !important;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown('<div class="hero-title">🌸 Employee Attrition Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Fill in employee details below to predict the likelihood of attrition</div>',
    unsafe_allow_html=True
)

if LOAD_ERROR:
    st.error(
        f"⚠️ Could not load model from `models/model.pkl`. "
        f"Make sure you run `streamlit run app.py` from the `aiml_project` root folder "
        f"(the one that contains the `models/` folder).\n\nDetails: {LOAD_ERROR}"
    )
    st.stop()


# INPUT FORM (TABS)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["👤 Personal", "💼 Job Details", "💰 Compensation", "⭐ Satisfaction & Growth", "📊 Dashboard"]
)

with tab1:
    with st.container(border=True):
        st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 60, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            home_dist = st.slider("Distance From Home (km)", 1, 29, 5)
        with c3:
            education = st.selectbox(
                "Education Level",
                [1, 2, 3, 4, 5],
                index=2,
                format_func=lambda x: {1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor",
                                        4: "4 - Master", 5: "5 - Doctor"}[x]
            )
            edu_field = st.selectbox(
                "Education Field",
                ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"]
            )

with tab2:
    with st.container(border=True):
        st.markdown('<div class="section-header">💼 Job Details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
            job_role = st.selectbox(
                "Job Role",
                ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                 "Healthcare Representative", "Manager", "Sales Representative", "Research Director",
                 "Human Resources"]
            )
        with c2:
            job_level = st.slider("Job Level", 1, 5, 2)
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        with c3:
            overtime = st.selectbox("OverTime", ["No", "Yes"])
            num_companies = st.slider("Number of Companies Worked At", 0, 9, 2)

with tab3:
    with st.container(border=True):
        st.markdown('<div class="section-header">💰 Compensation</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000, step=100)
            daily_rate = st.slider("Daily Rate", 100, 1500, 800)
        with c2:
            monthly_rate = st.slider("Monthly Rate", 2000, 27000, 14000, step=100)
            hourly_rate = st.slider("Hourly Rate", 30, 100, 65)
        with c3:
            salary_hike = st.slider("Percent Salary Hike (%)", 11, 25, 15)
            stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])

with tab4:
    with st.container(border=True):
        st.markdown('<div class="section-header">⭐ Satisfaction & Growth</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        satisfaction_map = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
        with c1:
            env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2,
                                             format_func=lambda x: satisfaction_map[x])
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2,
                                             format_func=lambda x: satisfaction_map[x])
            relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2,
                                                       format_func=lambda x: satisfaction_map[x])
        with c2:
            job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2,
                                            format_func=lambda x: satisfaction_map[x])
            work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4], index=2,
                                              format_func=lambda x: satisfaction_map[x])
            performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4], index=2,
                                               format_func=lambda x: satisfaction_map[x])
        with c3:
            training_times = st.slider("Training Times Last Year", 0, 6, 2)
            total_working_years = st.slider("Total Working Years", 0, 40, 8)
            years_at_company = st.slider("Years At Company", 0, 40, 5)

        c4, c5, c6 = st.columns(3)
        with c4:
            years_in_role = st.slider("Years In Current Role", 0, 18, 3)
        with c5:
            years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        with c6:
            years_with_manager = st.slider("Years With Current Manager", 0, 17, 3)

with tab5:
    dash_col1, dash_col2 = st.columns([1, 1])
    with dash_col1:
        st.markdown('<div class="section-header">📊 HR Insights & Model Performance</div>', unsafe_allow_html=True)
    with dash_col2:
        dash_view = st.radio(
            "View", ["🌈 HR Insights", "🎯 Model Performance"],
            horizontal=True, label_visibility="collapsed"
        )

    PASTEL = ["#f9a8d4", "#a5b4fc", "#6ee7b7", "#fde68a", "#fca5a5", "#93c5fd"]

    
    # HR INSIGHTS
    
    if dash_view == "🌈 HR Insights":
        try:
            hr_raw = load_hr_data()
            hr = decode_dummies(hr_raw)

            total = len(hr)
            attrition_count = int((hr["Attrition"] == 1).sum())
            attrition_rate = attrition_count / total * 100

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Employees", f"{total}")
            k2.metric("Attrition Cases", f"{attrition_count}")
            k3.metric("Attrition Rate", f"{attrition_rate:.1f}%")
            k4.metric("Avg Monthly Income", f"${hr['MonthlyInc'].mean():,.0f}" if "MonthlyInc" in hr else "N/A")

            st.markdown("<br>", unsafe_allow_html=True)
            r1c1, r1c2 = st.columns(2)

            with r1c1:
                pie_df = hr["AttritionLabel"].value_counts().reset_index()
                pie_df.columns = ["Attrition", "Count"]
                fig = px.pie(pie_df, names="Attrition", values="Count", hole=0.55,
                             color="Attrition", color_discrete_map={"No": "#6ee7b7", "Yes": "#fca5a5"},
                             title="Overall Attrition Split")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with r1c2:
                dept_df = hr.groupby(["Department", "AttritionLabel"]).size().reset_index(name="Count")
                fig = px.bar(dept_df, x="Department", y="Count", color="AttritionLabel", barmode="group",
                             color_discrete_map={"No": "#a5b4fc", "Yes": "#f9a8d4"},
                             title="Attrition by Department")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                fig = px.histogram(hr, x="Age", color="AttritionLabel", barmode="overlay", nbins=20,
                                    color_discrete_map={"No": "#93c5fd", "Yes": "#fca5a5"},
                                    title="Age Distribution by Attrition", opacity=0.75)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with r2c2:
                if "MonthlyInc" in hr:
                    fig = px.box(hr, x="AttritionLabel", y="MonthlyInc", color="AttritionLabel",
                                 color_discrete_map={"No": "#6ee7b7", "Yes": "#fca5a5"},
                                 title="Monthly Income vs Attrition")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

            r3c1, r3c2 = st.columns(2)
            with r3c1:
                ot_df = hr.groupby(["OverTime", "AttritionLabel"]).size().reset_index(name="Count")
                fig = px.bar(ot_df, x="OverTime", y="Count", color="AttritionLabel", barmode="group",
                             color_discrete_map={"No": "#a5b4fc", "Yes": "#f9a8d4"},
                             title="Attrition by OverTime")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with r3c2:
                if "Work_Life_Bal" in hr:
                    wlb_df = hr.groupby(["Work_Life_Bal", "AttritionLabel"]).size().reset_index(name="Count")
                    fig = px.bar(wlb_df, x="Work_Life_Bal", y="Count", color="AttritionLabel", barmode="group",
                                 color_discrete_map={"No": "#6ee7b7", "Yes": "#fca5a5"},
                                 title="Attrition by Work-Life Balance (1=Low, 4=High)")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

        except FileNotFoundError:
            st.warning(
                "⚠️ Couldn't find `data/hr.csv`. Make sure you're running `streamlit run app.py` "
                "from the `aiml_project` root folder."
            )
        except Exception as e:
            st.error(f"⚠️ Couldn't build HR insights: {e}")

    
    # MODEL PERFORMANCE
   
    else:
        try:
            X_test, y_test = load_test_data()
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, proba)

            st.info(
                "ℹ️ These metrics are computed live from `data/hr.csv` (rebuilt with the exact same "
                "80/20 stratified split, `random_state=42`, used in `model_training.ipynb`) and "
                "`models/model.pkl`."
            )

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{acc*100:.1f}%")
            m2.metric("Precision", f"{prec*100:.1f}%")
            m3.metric("Recall", f"{rec*100:.1f}%")
            m4.metric("F1 Score", f"{f1*100:.1f}%")
            m5.metric("AUC", f"{auc:.3f}")

            st.markdown("<br>", unsafe_allow_html=True)
            p1, p2 = st.columns(2)

            with p1:
                cm = confusion_matrix(y_test, preds)
                fig = px.imshow(
                    cm, text_auto=True, x=["Predicted: Stay", "Predicted: Leave"],
                    y=["Actual: Stay", "Actual: Leave"],
                    color_continuous_scale=["#fdf2f8", "#a78bfa"],
                    title="Confusion Matrix"
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with p2:
                fpr, tpr, _ = roc_curve(y_test, proba)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})",
                                          line=dict(color="#f472b6", width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                          line=dict(color="#c4b5fd", dash="dash")))
                fig.update_layout(
                    title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if hasattr(model, "coef_"):
                importance = pd.DataFrame({
                    "Feature": model.feature_names_in_,
                    "Importance": np.abs(model.coef_[0])
                }).sort_values("Importance", ascending=True).tail(15)
                fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale=["#fbcfe8", "#7c3aed"],
                             title="Top 15 Feature Importances (|coefficient|)")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        except FileNotFoundError:
            st.warning(
                "⚠️ Couldn't find `data/hr.csv`. Make sure you're running "
                "`streamlit run app.py` from the `aiml_project` root folder."
            )
        except Exception as e:
            st.error(f"⚠️ Couldn't compute model performance: {e}")

st.markdown("<br>", unsafe_allow_html=True)



# ENCODING LOGIC (must mirror pd.get_dummies(drop_first=True) exactly)

def build_feature_vector():
    row = {
        'Age': age,
        'DailyRate': daily_rate,
        'HomeDist': home_dist,
        'Education': education,
        'EnvironmentSatisfaction': env_satisfaction,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLvl': job_level,
        'JobSatisfaction': job_satisfaction,
        'MonthlyInc': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies,
        'SalaryHike': salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times,
        'Work_Life_Bal': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_role,
        'YearsSinceLastPromotion': years_since_promotion,
        'YearsWithCurrManager': years_with_manager,

        # BusinessTravel (baseline dropped: Non-Travel)
        'Travel_Travel_Frequently': 1 if business_travel == "Travel_Frequently" else 0,
        'Travel_Travel_Rarely': 1 if business_travel == "Travel_Rarely" else 0,

        # Department (baseline dropped: Human Resources)
        'Dept_Research & Development': 1 if department == "Research & Development" else 0,
        'Dept_Sales': 1 if department == "Sales" else 0,

        # EducationField (baseline dropped: Human Resources)
        'EduField_Life Sciences': 1 if edu_field == "Life Sciences" else 0,
        'EduField_Marketing': 1 if edu_field == "Marketing" else 0,
        'EduField_Medical': 1 if edu_field == "Medical" else 0,
        'EduField_Other': 1 if edu_field == "Other" else 0,
        'EduField_Technical Degree': 1 if edu_field == "Technical Degree" else 0,

        # Gender (baseline dropped: Female)
        'Gender_Male': 1 if gender == "Male" else 0,

        # JobRole (baseline dropped: Healthcare Representative)
        'JobRole_Human Resources': 1 if job_role == "Human Resources" else 0,
        'JobRole_Laboratory Technician': 1 if job_role == "Laboratory Technician" else 0,
        'JobRole_Manager': 1 if job_role == "Manager" else 0,
        'JobRole_Manufacturing Director': 1 if job_role == "Manufacturing Director" else 0,
        'JobRole_Research Director': 1 if job_role == "Research Director" else 0,
        'JobRole_Research Scientist': 1 if job_role == "Research Scientist" else 0,
        'JobRole_Sales Executive': 1 if job_role == "Sales Executive" else 0,
        'JobRole_Sales Representative': 1 if job_role == "Sales Representative" else 0,

        # MaritalStatus (baseline dropped: Divorced)
        'Marital_Married': 1 if marital == "Married" else 0,
        'Marital_Single': 1 if marital == "Single" else 0,

        # OverTime (baseline dropped: No)
        'OverTime_Yes': 1 if overtime == "Yes" else 0,
    }
    vector = np.array([[row[col] for col in FEATURE_ORDER]], dtype=float)
    return vector


# PREDICT BUTTON

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_b:
    predict_clicked = st.button("✨ Predict Attrition Risk", use_container_width=True)

if predict_clicked:
    with st.spinner("Analyzing employee profile..."):
        time.sleep(0.6)
        X = build_feature_vector()
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        leave_prob = float(proba[1]) * 100
        stay_prob = float(proba[0]) * 100

    st.markdown("<br>", unsafe_allow_html=True)
    result_col1, result_col2 = st.columns([1.1, 1])

    with result_col1:
        if pred == 1:
            st.markdown(f"""
            <div class="result-card leave-card">
                <span class="result-emoji">⚠️</span>
                <div class="result-heading" style="color:#b91c1c;">High Attrition Risk</div>
                <div class="result-sub">This employee is likely to leave the company</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{leave_prob:.1f}%; background: linear-gradient(90deg, #fb7185, #f97316);"></div>
                </div>
                <div style="margin-top:0.6rem; font-weight:600; color:#b91c1c; font-size:1.3rem;">
                    {leave_prob:.1f}% likelihood of leaving
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card stay-card">
                <span class="result-emoji">🌿</span>
                <div class="result-heading" style="color:#047857;">Low Attrition Risk</div>
                <div class="result-sub">This employee is likely to stay with the company</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{stay_prob:.1f}%; background: linear-gradient(90deg, #34d399, #60a5fa);"></div>
                </div>
                <div style="margin-top:0.6rem; font-weight:600; color:#047857; font-size:1.3rem;">
                    {stay_prob:.1f}% likelihood of staying
                </div>
            </div>
            """, unsafe_allow_html=True)

    with result_col2:
        st.markdown('<div class="section-header">📊 Prediction Breakdown</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Stay Probability", f"{stay_prob:.1f}%")
        m2.metric("Leave Probability", f"{leave_prob:.1f}%")
        st.progress(int(leave_prob))
        if pred == 1:
            st.info("💡 Consider reviewing compensation, workload, and career growth opportunities for this profile.")
        else:
            st.success("💡 This profile shows healthy retention indicators.")

    if pred == 0:
        st.balloons()

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#9ca3af; font-size:0.85rem;">💜  Employee Attrition Prediction Model 💜</div>',
    unsafe_allow_html=True
)
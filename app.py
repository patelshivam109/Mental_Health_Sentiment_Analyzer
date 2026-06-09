from pathlib import Path
from html import escape
import re

import joblib
import pandas as pd
import streamlit as st
from textblob import TextBlob


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
REPORTS_DIR = ROOT / "reports"
VISUALS_DIR = ROOT / "visuals"
MODELS_DIR = ROOT / "models"


st.set_page_config(
    page_title="Mental Health Sentiment Analyzer",
    page_icon="MH",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --app-bg: #f5f7fb;
        --panel-bg: #ffffff;
        --panel-soft: #f8fafc;
        --text-main: #17212b;
        --text-muted: #617080;
        --border: #d9e1ea;
        --brand: #2f6f9f;
        --brand-strong: #255a82;
        --green: #247a52;
        --red: #b9444b;
        --amber: #9a6a10;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --app-bg: #0e1117;
            --panel-bg: #171b22;
            --panel-soft: #202631;
            --text-main: #f4f7fb;
            --text-muted: #aab6c4;
            --border: #303846;
            --brand: #76b7e5;
            --brand-strong: #9bcaf0;
            --green: #58c58e;
            --red: #ff7f87;
            --amber: #f0c36a;
        }
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(47, 111, 159, 0.10), transparent 34rem),
            var(--app-bg);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--text-main);
    }
    p, li, label, div {
        letter-spacing: 0;
    }
    [data-testid="stMetric"] {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stMetric"] * {
        color: var(--text-main) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102030 0%, #14283a 58%, #182536 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f6f8fb !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 7px;
        padding: 0.3rem 0.4rem;
        margin-bottom: 0.15rem;
    }
    .hero {
        background: linear-gradient(135deg, #18344d 0%, #2f6f9f 56%, #3f8b6b 100%);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 10px;
        padding: 26px 30px;
        color: #ffffff;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.2rem;
    }
    .hero-kicker {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.86;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .hero-title {
        font-size: clamp(1.9rem, 3vw, 3rem);
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0;
    }
    .hero-copy {
        max-width: 820px;
        margin-top: 0.75rem;
        color: rgba(255,255,255,0.90);
        font-size: 1.02rem;
    }
    .sidebar-brand {
        padding: 0.35rem 0.25rem 1rem;
    }
    .sidebar-logo {
        width: 42px;
        height: 42px;
        border-radius: 8px;
        background: #ffffff;
        color: #18344d !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        margin-bottom: 0.65rem;
    }
    .sidebar-title {
        font-size: 1.02rem;
        font-weight: 800;
        line-height: 1.15;
    }
    .sidebar-subtitle {
        color: rgba(255,255,255,0.72) !important;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }
    .kpi-card {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 18px 18px 16px;
        min-height: 124px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }
    .kpi-label {
        color: var(--text-muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        color: var(--text-main);
        font-size: 2rem;
        line-height: 1.05;
        font-weight: 800;
    }
    .kpi-caption {
        color: var(--text-muted);
        font-size: 0.88rem;
        margin-top: 0.5rem;
    }
    .section-title {
        margin: 1.35rem 0 0.55rem;
    }
    .section-title h2 {
        font-size: 1.28rem;
        margin: 0;
        font-weight: 800;
    }
    .section-title p {
        margin: 0.25rem 0 0;
        color: var(--text-muted);
        font-size: 0.95rem;
    }
    .panel {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 18px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .insight-list {
        display: grid;
        gap: 0.72rem;
    }
    .insight {
        border-left: 4px solid var(--brand);
        background: var(--panel-soft);
        color: var(--text-main);
        border-radius: 7px;
        padding: 12px 14px;
        font-size: 0.95rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        background: rgba(47, 111, 159, 0.12);
        color: var(--brand-strong);
        border: 1px solid rgba(47, 111, 159, 0.22);
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 0.65rem;
    }
    .callout {
        padding: 13px 15px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--panel-soft);
        color: var(--text-main);
        margin: 0.25rem 0 1rem;
    }
    .status-note {
        padding: 13px 15px;
        border-left: 4px solid #2f6f9f;
        background: var(--panel-soft);
        color: var(--text-main);
        border-radius: 7px;
        margin: 0.25rem 0 1rem;
    }
    .risk-note {
        padding: 13px 15px;
        border-left: 4px solid #c44e52;
        background: rgba(185, 68, 75, 0.10);
        color: var(--text-main);
        border-radius: 7px;
        margin: 0.25rem 0 1rem;
    }
    .prediction-card {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.07);
    }
    .small-muted {
        color: var(--text-muted);
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = pd.read_csv(DATA_DIR / "cleaned_survey.csv")
    comments = pd.read_csv(DATA_DIR / "processed_comments_sentiment.csv")
    return cleaned, comments


@st.cache_data
def load_report_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / filename)


@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "sentiment_tfidf_pipeline.pkl")


def image(path: str, caption: str | None = None) -> None:
    st.image(str(VISUALS_DIR / path), caption=caption, use_container_width=True)


def pct(value: float) -> str:
    return f"{value:.1f}%"


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Employee Wellness Intelligence</div>
            <div class="hero-title">Mental Health Sentiment Analyzer</div>
            <div class="hero-copy">
                A professional NLP dashboard for survey sentiment, model performance,
                at-risk cohort discovery, and HR-ready wellness recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-title">
            <h2>{escape(title)}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value">{escape(value)}</div>
            <div class="kpi-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_panel(items: list[str]) -> None:
    rows = "\n".join(f'<div class="insight">{escape(item)}</div>' for item in items)
    st.markdown(f'<div class="panel"><div class="insight-list">{rows}</div></div>', unsafe_allow_html=True)


def status_pill(text: str) -> None:
    st.markdown(f'<div class="status-pill">{escape(text)}</div>', unsafe_allow_html=True)


def final_sentiment_output(text: str, model_prediction: str) -> tuple[str, float, list[str]]:
    polarity = TextBlob(text).sentiment.polarity
    lowered = text.lower()
    risk_terms = [
        "anxious",
        "anxiety",
        "burnout",
        "burned out",
        "concern",
        "concerned",
        "depressed",
        "depression",
        "distress",
        "distressed",
        "overwhelmed",
        "problem",
        "problems",
        "sad",
        "stress",
        "stressed",
        "unsupported",
        "panic",
        "exhausted",
        "hopeless",
        "hate",
        "issue",
        "issues",
        "lonely",
        "alone",
        "worry",
        "worried",
    ]
    risk_phrases = [
        "trust issue",
        "trust issues",
        "mental health issue",
        "mental health issues",
        "not comfortable",
        "do not trust",
        "don't trust",
    ]
    matched_terms = [term for term in risk_terms if re.search(rf"\b{re.escape(term)}\b", lowered)]
    matched_terms.extend([phrase for phrase in risk_phrases if phrase in lowered])

    if polarity < -0.05 or matched_terms:
        return "Negative", polarity, matched_terms
    if polarity > 0.05:
        return "Positive", polarity, matched_terms
    return model_prediction, polarity, matched_terms


cleaned_df, comments_df = load_data()
model = load_model()
model_results = load_report_csv("sentiment_model_results.csv")
sentiment_dist = load_report_csv("sentiment_distribution.csv")


st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-logo">MH</div>
        <div class="sidebar-title">Mental Health<br>Sentiment Analyzer</div>
        <div class="sidebar-subtitle">NLP dashboard for HR wellness insights</div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Sentiment Analysis",
        "At-Risk Cohorts",
        "Model Performance",
        "Predict Sentiment",
        "Reports",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Dataset Snapshot**")
st.sidebar.write(f"Survey rows: {len(cleaned_df):,}")
st.sidebar.write(f"Comment rows: {len(comments_df):,}")
st.sidebar.write("Pipeline: TF-IDF + Logistic Regression")


hero()


if page == "Executive Overview":
    positive = int(sentiment_dist.loc[sentiment_dist["Sentiment"] == "Positive", "Count"].sum())
    negative = int(sentiment_dist.loc[sentiment_dist["Sentiment"] == "Negative", "Count"].sum())
    neutral = int(sentiment_dist.loc[sentiment_dist["Sentiment"] == "Neutral", "Count"].sum())
    treatment_yes = int((cleaned_df["treatment"] == "Yes").sum())

    status_pill("Executive Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Survey Responses", f"{len(cleaned_df):,}", "Cleaned employee survey records")
    with col2:
        kpi_card("Usable Comments", f"{len(comments_df):,}", "Free-text rows used for NLP")
    with col3:
        kpi_card("Treatment Yes", f"{treatment_yes:,}", pct(treatment_yes / len(cleaned_df) * 100) + " of cleaned records")
    with col4:
        kpi_card("Negative Comments", f"{negative:,}", pct(negative / len(comments_df) * 100) + " of comment rows")

    section("Key Findings", "Signals that matter most for workplace wellness review.")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        insight_panel(
            [
                "Employees with family history show higher treatment-seeking behavior.",
                "Work interference is one of the clearest high-risk indicators.",
                "Comment sentiment is mixed, with positive comments highest but a meaningful negative group present.",
                "The comment sample is limited, so model output should support review rather than replace human judgment.",
            ]
        )
    with c2:
        image("sentiment_distribution.png", "Sentiment distribution from processed comments")

    section("Workplace Mental Health Trends", "Core survey patterns used by the dashboard.")
    a, b = st.columns(2)
    with a:
        image("treatment_distribution.png", "Treatment distribution")
    with b:
        image("work_interfere_distribution.png", "Work interference by treatment")


elif page == "Sentiment Analysis":
    status_pill("Sentiment Analysis")
    section("Sentiment Distribution", "Breakdown of positive, neutral, and negative employee comments.")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.dataframe(sentiment_dist, use_container_width=True, hide_index=True)
        st.bar_chart(sentiment_dist.set_index("Sentiment")["Count"])
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        image("sentiment_distribution.png")

    section("Sentiment Across Employee Groups", "Compare sentiment proportions by demographics and location.")
    gender_sentiment = load_report_csv("sentiment_by_gender.csv")
    country_sentiment = load_report_csv("sentiment_by_country_top10.csv")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**By gender**")
        st.dataframe(gender_sentiment, use_container_width=True, hide_index=True)
    with g2:
        st.markdown("**Top countries by negative sentiment ratio**")
        st.dataframe(country_sentiment, use_container_width=True, hide_index=True)

    section("Comment Explorer", "Filter and inspect employee comments used in sentiment analysis.")
    sentiment_filter = st.multiselect(
        "Sentiment",
        ["Negative", "Neutral", "Positive"],
        default=["Negative", "Neutral", "Positive"],
    )
    filtered = comments_df[comments_df["sentiment"].isin(sentiment_filter)]
    st.dataframe(
        filtered[["Age", "Gender", "Country", "work_interfere", "treatment", "sentiment", "comments"]]
        .head(100),
        use_container_width=True,
        hide_index=True,
    )


elif page == "At-Risk Cohorts":
    status_pill("Risk Analytics")
    section("At-Risk Cohort Indicators", "Survey-based groups that may require stronger wellness support.")
    st.markdown(
        '<div class="risk-note">High-risk groups are identified from treatment status, work interference, family history, age group, country, and negative comment sentiment. Role analysis is not included because the source dataset has no role/job-title column.</div>',
        unsafe_allow_html=True,
    )

    age = load_report_csv("treatment_by_age_group.csv")
    family = load_report_csv("treatment_by_family_history.csv")
    work = load_report_csv("treatment_by_work_interference.csv")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Treatment by age group**")
        st.dataframe(age, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Treatment by family history**")
        st.dataframe(family, use_container_width=True, hide_index=True)
    with c3:
        st.markdown("**Treatment by work interference**")
        st.dataframe(work, use_container_width=True, hide_index=True)

    section("Supporting Visualizations", "Demographic and workplace factors connected to wellness patterns.")
    v1, v2 = st.columns(2)
    with v1:
        image("family_history_distribution.png")
    with v2:
        image("age_distribution.png")

    v3, v4 = st.columns(2)
    with v3:
        image("country_distribution.png")
    with v4:
        image("gender_distribution.png")


elif page == "Model Performance":
    status_pill("Model Evaluation")
    section("Model Comparison", "Classifier performance across sentiment labels.")
    st.dataframe(model_results, use_container_width=True, hide_index=True)

    best = model_results.sort_values(["Weighted F1", "Accuracy"], ascending=False).iloc[0]
    m1, m2, m3 = st.columns(3)
    with m1:
        kpi_card("Selected Model", str(best["Model"]), "Highest Weighted F1 score")
    with m2:
        kpi_card("Accuracy", f"{best['Accuracy']:.3f}", "Held-out test split")
    with m3:
        kpi_card("Weighted F1", f"{best['Weighted F1']:.3f}", "Primary selection metric")

    section("Performance Visuals", "Model comparison and confusion matrix for the selected classifier.")
    c1, c2 = st.columns(2)
    with c1:
        image("sentiment_model_comparison.png")
    with c2:
        image("sentiment_confusion_matrix.png")

    st.markdown(
        '<div class="status-note">Logistic Regression was selected because it achieved the highest Weighted F1 score, which is a better selection metric than accuracy alone for this multi-class sentiment task.</div>',
        unsafe_allow_html=True,
    )


elif page == "Predict Sentiment":
    status_pill("Live Prediction")
    section("Analyze New Employee Feedback", "Enter a short employee comment and review the model output with wellness safeguards.")
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    text = st.text_area(
        "Employee comment",
        height=180,
        placeholder="Example: I feel anxious and overwhelmed because of work pressure.",
    )
    predict_clicked = st.button("Predict Sentiment", type="primary", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        if text.strip():
            model_prediction = model.predict([text.strip()])[0]
            prediction, polarity, matched_terms = final_sentiment_output(text.strip(), model_prediction)
            p1, p2, p3 = st.columns(3)
            with p1:
                kpi_card("Final Sentiment", prediction, "Output shown to reviewer")
            with p2:
                kpi_card("Model Prediction", model_prediction, "Raw TF-IDF classifier result")
            with p3:
                kpi_card("Polarity Score", f"{polarity:.3f}", "TextBlob polarity signal")

            if matched_terms:
                st.markdown(
                    '<div class="risk-note">Detected wellness risk terms: '
                    + escape(", ".join(matched_terms))
                    + "</div>",
                    unsafe_allow_html=True,
                )

            if prediction == "Negative":
                st.warning("This comment may need HR wellness review.")
            elif prediction == "Positive":
                st.success("This comment appears positive based on the sentiment analysis workflow.")
            else:
                st.info("This comment appears neutral based on the sentiment analysis workflow.")
        else:
            st.error("Please enter a comment before predicting.")

    st.markdown(
        '<div class="callout small-muted">Model output is for project analysis only and is not a diagnosis or employment decision tool.</div>',
        unsafe_allow_html=True,
    )


elif page == "Reports":
    status_pill("Reports")
    section("Documentation and Reports", "Open final documentation and supporting project reports inside the dashboard.")
    report_files = {
        "Sentiment Analysis Report": "sentiment_analysis_report.md",
        "At-Risk Cohort Report": "at_risk_cohort_report.md",
        "HR Recommendations": "hr_recommendations.md",
        "Final Documentation Report": "final_documentation_report.md",
    }
    selected_report = st.selectbox("Report", list(report_files.keys()))
    report_path = REPORTS_DIR / report_files[selected_report]
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("This report file has not been generated yet.")

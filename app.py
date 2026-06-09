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
        --app-bg: #f6f3ee;
        --panel-bg: #ffffff;
        --panel-soft: #faf7f2;
        --text-main: #18212b;
        --text-muted: #64707c;
        --border: #d8d2c8;
        --brand: #1f4e5f;
        --brand-strong: #163947;
        --brand-soft: #d9e7eb;
        --accent: #6c8c6d;
        --accent-soft: #e6f0e6;
        --warm: #a06b46;
        --warm-soft: #f3e7dc;
        --rose: #a84a5a;
        --rose-soft: #f7e7ea;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --app-bg: #11161c;
            --panel-bg: #171d24;
            --panel-soft: #1d242d;
            --text-main: #f3f7fb;
            --text-muted: #a7b3bf;
            --border: #2d3844;
            --brand: #8dc2cc;
            --brand-strong: #b8e0e6;
            --brand-soft: rgba(141, 194, 204, 0.12);
            --accent: #84b38a;
            --accent-soft: rgba(132, 179, 138, 0.12);
            --warm: #d2a57b;
            --warm-soft: rgba(210, 165, 123, 0.12);
            --rose: #d77b8a;
            --rose-soft: rgba(215, 123, 138, 0.12);
        }
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(31, 78, 95, 0.12), transparent 32rem),
            radial-gradient(circle at bottom right, rgba(160, 107, 70, 0.08), transparent 26rem),
            var(--app-bg);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1420px;
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
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stMetric"] * {
        color: var(--text-main) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13212b 0%, #152a35 55%, #182834 100%);
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
        background:
            linear-gradient(135deg, rgba(24,49,60,0.96) 0%, rgba(31,78,95,0.96) 54%, rgba(160,107,70,0.94) 100%);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 24px 28px;
        color: #ffffff;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.20);
        margin-bottom: 1rem;
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
        border-radius: 12px;
        padding: 18px 18px 16px;
        min-height: 124px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
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
        margin: 1.15rem 0 0.5rem;
    }
    .section-title h2 {
        font-size: 1.18rem;
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
        border-radius: 12px;
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
        border-radius: 9px;
        padding: 12px 14px;
        font-size: 0.95rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        background: var(--brand-soft);
        color: var(--brand-strong);
        border: 1px solid rgba(31, 78, 95, 0.18);
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
        border-left: 4px solid var(--brand);
        background: var(--panel-soft);
        color: var(--text-main);
        border-radius: 9px;
        margin: 0.25rem 0 1rem;
    }
    .risk-note {
        padding: 13px 15px;
        border-left: 4px solid var(--rose);
        background: var(--rose-soft);
        color: var(--text-main);
        border-radius: 9px;
        margin: 0.25rem 0 1rem;
    }
    .prediction-card {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.07);
    }
    .small-muted {
        color: var(--text-muted);
        font-size: 0.86rem;
    }
    .report-card {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.9rem;
    }
    .report-title {
        font-weight: 800;
        color: var(--text-main);
        margin-bottom: 0.25rem;
    }
    .report-meta {
        color: var(--text-muted);
        font-size: 0.88rem;
    }
    .thin-rule {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 1rem 0;
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


def report_card(label: str, description: str, status: str = "Ready") -> None:
    st.markdown(
        f"""
        <div class="report-card">
            <div class="report-title">{escape(label)}</div>
            <div class="report-meta">{escape(description)}</div>
            <div class="thin-rule"></div>
            <div class="report-meta"><strong>Status:</strong> {escape(status)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    if polarity > 0.15:
        return "Positive", polarity, matched_terms
    return "Neutral", polarity, matched_terms


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
        "HR Recommendations",
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
        st.markdown(
            '<div class="small-muted">The distribution helps verify whether comments lean positive, neutral, or negative before model use.</div>',
            unsafe_allow_html=True,
        )

    section("Sentiment Across Employee Groups", "Compare sentiment proportions by demographics and location.")
    gender_sentiment = load_report_csv("sentiment_by_gender.csv")
    country_sentiment = load_report_csv("sentiment_by_country_top10.csv")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**By gender**")
        st.dataframe(gender_sentiment, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Top countries by negative sentiment ratio**")
        st.dataframe(country_sentiment, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    section("Comment Explorer", "Filter and inspect employee comments used in sentiment analysis.")
    sentiment_filter = st.multiselect(
        "Sentiment",
        ["Negative", "Neutral", "Positive"],
        default=["Negative", "Neutral", "Positive"],
    )
    filtered = comments_df[comments_df["sentiment"].isin(sentiment_filter)]
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.dataframe(
        filtered[["Age", "Gender", "Country", "work_interfere", "treatment", "sentiment", "comments"]]
        .head(100),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


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
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Treatment by age group**")
        st.dataframe(age, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Treatment by family history**")
        st.dataframe(family, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Treatment by work interference**")
        st.dataframe(work, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.dataframe(model_results, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

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


elif page == "HR Recommendations":
    status_pill("HR Guidance")
    section(
        "HR Recommendations",
        "Actionable guidance derived from treatment status, family history, work interference, and comment sentiment.",
    )

    st.markdown(
        '<div class="panel">'
        '<div class="report-title">Recommended Actions</div>'
        '<div class="thin-rule"></div>'
        '<div class="insight-list">'
        '<div class="insight">Provide targeted mental health support for employees reporting frequent work interference.</div>'
        '<div class="insight">Increase visibility of confidential help channels, anonymity settings, and care options.</div>'
        '<div class="insight">Offer proactive check-ins for employees with family history risk factors or repeated negative sentiment.</div>'
        '<div class="insight">Monitor high-risk age groups and repeated stress signals, but avoid making individual decisions from model output alone.</div>'
        '<div class="insight">Use survey trends as a wellness signal, not a diagnostic or performance-management tool.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("**Priority actions for HR**")
        st.markdown(
            """
            1. Improve access to confidential mental health support and clear escalation paths.
            2. Reassure employees about anonymity and non-retaliation in wellness surveys.
            3. Schedule structured check-ins for teams with high work-interference rates.
            4. Strengthen manager awareness around early signs of burnout, stress, and disengagement.
            5. Review policy visibility so employees can quickly find benefits and support options.
            """
        )
    with right:
        st.markdown("**Supporting evidence**")
        st.markdown(
            """
            - Family history shows a strong relationship with treatment-seeking behavior.
            - Work interference is one of the strongest risk signals in this dataset.
            - Negative comments should be reviewed as wellness indicators, not isolated complaints.
            - HR should focus on support and trend monitoring rather than individual inference.
            """
        )

    section("Related Report", "The same recommendations are also documented in the final report.")
    report_path = REPORTS_DIR / "hr_recommendations.md"
    if report_path.exists():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(report_path.read_text(encoding="utf-8"))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("HR recommendations report is not available yet.")


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
    files = [
        ("Documentation", "Word document and markdown copy of the final report."),
        ("Model", "Trained TF-IDF sentiment pipeline."),
        ("App", "Streamlit application entry point."),
        ("Data", "Cleaned and processed datasets used by the app."),
    ]
    cols = st.columns(4)
    for col, (label, desc) in zip(cols, files):
        with col:
            report_card(label, desc, "Included")
    if report_path.exists():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(report_path.read_text(encoding="utf-8"))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("This report file has not been generated yet.")

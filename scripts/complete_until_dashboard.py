from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from textblob import TextBlob

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
VISUALS_DIR = ROOT / "visuals"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

for directory in [VISUALS_DIR, REPORTS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)


def clean_text(text: object) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentiment_label(text: object) -> str:
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.05:
        return "Positive"
    if polarity < -0.05:
        return "Negative"
    return "Neutral"


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def markdown_table(data: pd.DataFrame) -> str:
    table = data.reset_index() if data.index.name or not isinstance(data.index, pd.RangeIndex) else data.copy()
    table = table.astype(str).replace("nan", "")
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(value) for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def create_eda_visuals(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 4.5))
    sns.histplot(df["Age"], bins=20, kde=True, color="#2f6f9f")
    plt.title("Age Distribution of Employees")
    plt.xlabel("Age")
    plt.ylabel("Count")
    save_plot("age_distribution.png")

    plt.figure(figsize=(7, 4.5))
    sns.countplot(
        data=df,
        x="Gender",
        hue="Gender",
        order=df["Gender"].value_counts().index,
        palette="Set2",
        legend=False,
    )
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    save_plot("gender_distribution.png")

    plt.figure(figsize=(6, 4.5))
    sns.countplot(
        data=df,
        x="treatment",
        hue="treatment",
        order=["No", "Yes"],
        palette="Set1",
        legend=False,
    )
    plt.title("Mental Health Treatment Distribution")
    plt.xlabel("Sought Treatment")
    plt.ylabel("Count")
    save_plot("treatment_distribution.png")

    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x="family_history", hue="treatment", palette="Set1")
    plt.title("Family History vs Treatment")
    plt.xlabel("Family History")
    plt.ylabel("Count")
    save_plot("family_history_distribution.png")

    plt.figure(figsize=(8, 4.5))
    order = ["Never", "Rarely", "Sometimes", "Often", "Don't know"]
    sns.countplot(data=df, x="work_interfere", hue="treatment", order=order, palette="Set1")
    plt.title("Work Interference vs Treatment")
    plt.xlabel("Work Interference")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    save_plot("work_interfere_distribution.png")

    top_countries = df["Country"].value_counts().head(10)
    plt.figure(figsize=(9, 4.8))
    top_countries.plot(kind="bar", color="#517a3f")
    plt.title("Top 10 Countries in Survey")
    plt.xlabel("Country")
    plt.ylabel("Responses")
    plt.xticks(rotation=35, ha="right")
    save_plot("country_distribution.png")


def train_sentiment_models(comments_df: pd.DataFrame):
    X = comments_df["clean_comments"]
    y = comments_df["sentiment"]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    models = {
        "Naive Bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
                ("model", MultinomialNB()),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
                ("model", LinearSVC(class_weight="balanced")),
            ]
        ),
    }

    rows = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Weighted F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            }
        )
        trained[name] = (model, predictions)

    param_grid = {
        "tfidf__max_features": [300, 500, 1000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "model__C": [0.5, 1.0, 2.0],
    }
    lr_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    cv = min(3, y_train.value_counts().min())
    if cv >= 2:
        grid = GridSearchCV(lr_pipeline, param_grid, cv=cv, scoring="f1_weighted")
        grid.fit(X_train, y_train)
        tuned_model = grid.best_estimator_
    else:
        tuned_model = trained["Logistic Regression"][0]

    tuned_predictions = tuned_model.predict(X_test)
    rows.append(
        {
            "Model": "Tuned Logistic Regression",
            "Accuracy": accuracy_score(y_test, tuned_predictions),
            "Weighted F1": f1_score(y_test, tuned_predictions, average="weighted", zero_division=0),
        }
    )

    results = (
        pd.DataFrame(rows)
        .sort_values(["Weighted F1", "Accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    best_name = results.iloc[0]["Model"]
    if best_name == "Tuned Logistic Regression":
        best_model = tuned_model
        best_predictions = tuned_predictions
    else:
        best_model, best_predictions = trained[best_name]

    report = classification_report(y_test, best_predictions, zero_division=0)
    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, best_predictions, labels=labels)

    return {
        "results": results,
        "best_name": best_name,
        "best_model": best_model,
        "classification_report": report,
        "confusion_matrix": matrix,
        "labels": labels,
        "test_size": len(y_test),
        "train_size": len(y_train),
    }


def create_sentiment_visuals(comments_df: pd.DataFrame, model_info: dict) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6.5, 4.5))
    sns.countplot(
        data=comments_df,
        x="sentiment",
        hue="sentiment",
        order=["Negative", "Neutral", "Positive"],
        palette={"Negative": "#c44e52", "Neutral": "#8172b3", "Positive": "#55a868"},
        legend=False,
    )
    plt.title("Sentiment Distribution in Employee Comments")
    plt.xlabel("Sentiment")
    plt.ylabel("Comments")
    save_plot("sentiment_distribution.png")

    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=model_info["results"], x="Weighted F1", y="Model", color="#4c72b0")
    plt.title("Sentiment Model Comparison")
    plt.xlabel("Weighted F1 Score")
    plt.ylabel("")
    plt.xlim(0, 1)
    save_plot("sentiment_model_comparison.png")

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        model_info["confusion_matrix"],
        annot=True,
        fmt="d",
        xticklabels=model_info["labels"],
        yticklabels=model_info["labels"],
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix: {model_info['best_name']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot("sentiment_confusion_matrix.png")


def build_reports(df: pd.DataFrame, comments_df: pd.DataFrame, model_info: dict) -> None:
    sentiment_counts = comments_df["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
    gender_sentiment = pd.crosstab(comments_df["Gender"], comments_df["sentiment"], normalize="index").round(3)
    country_sentiment = (
        pd.crosstab(comments_df["Country"], comments_df["sentiment"], normalize="index")
        .round(3)
        .sort_values(by="Negative", ascending=False, na_position="last")
        .head(10)
    )

    treatment_by_family = pd.crosstab(df["family_history"], df["treatment"])
    treatment_by_work = pd.crosstab(df["work_interfere"], df["treatment"])
    treatment_by_age = pd.crosstab(df["Age_Group"], df["treatment"])
    negative_by_gender = pd.crosstab(comments_df["Gender"], comments_df["sentiment"])

    model_info["results"].to_csv(REPORTS_DIR / "sentiment_model_results.csv", index=False)
    sentiment_counts.to_csv(REPORTS_DIR / "sentiment_distribution.csv", index=False)
    gender_sentiment.to_csv(REPORTS_DIR / "sentiment_by_gender.csv")
    country_sentiment.to_csv(REPORTS_DIR / "sentiment_by_country_top10.csv")
    treatment_by_family.to_csv(REPORTS_DIR / "treatment_by_family_history.csv")
    treatment_by_work.to_csv(REPORTS_DIR / "treatment_by_work_interference.csv")
    treatment_by_age.to_csv(REPORTS_DIR / "treatment_by_age_group.csv")

    sentiment_report = f"""# Sentiment Analysis Report

## Scope

This report covers the project work up to the dashboard stage from the internship PDF. The source dataset does not contain ground-truth sentiment labels, so comment sentiment labels were generated with TextBlob polarity and then used to train TF-IDF based classifiers.

## Data Used

- Total cleaned survey rows: {len(df)}
- Rows with usable employee comments: {len(comments_df)}
- Train rows: {model_info["train_size"]}
- Test rows: {model_info["test_size"]}

## Sentiment Label Distribution

{markdown_table(sentiment_counts)}

## Model Comparison

{markdown_table(model_info["results"].round(4))}

## Selected Model

Best model selected by weighted F1 score: **{model_info["best_name"]}**.

## Classification Report

```text
{model_info["classification_report"]}
```

## Notes

Because the sentiment labels are automatically derived rather than manually annotated, this model should be treated as an internship/project NLP workflow artifact, not a clinical or HR decision system.
"""

    risk_report = f"""# At-Risk Cohort Identification Report

## Basis

At-risk cohorts were identified using survey signals from the PDF task: treatment seeking, work interference, family history, age group, location, and negative sentiment in free-text comments. The dataset does not include a role/job-title column, so role-based cohort analysis could not be performed exactly.

## Treatment by Family History

{markdown_table(treatment_by_family)}

## Treatment by Work Interference

{markdown_table(treatment_by_work)}

## Treatment by Age Group

{markdown_table(treatment_by_age)}

## Comment Sentiment by Gender

{markdown_table(negative_by_gender)}

## Highest Negative-Sentiment Countries Among Comment Rows

{markdown_table(country_sentiment)}

## Key At-Risk Groups

- Employees reporting that mental health often or sometimes interferes with work.
- Employees with a family history of mental illness.
- Employees in age groups with higher treatment-seeking rates, especially 36-45 and 46-70 in this dataset.
- Employee comment groups with higher negative sentiment rates.
"""

    hr_report = """# HR Insights and Recommendations Report

## Recommendations

- Provide targeted support for employees who report frequent work interference.
- Improve awareness of mental health benefits, care options, anonymity, and help-seeking channels.
- Create confidential check-in options for employees with family-history risk factors or repeated workplace stress signals.
- Monitor survey trends by age, gender, country, family history, and work interference before drawing individual-level conclusions.
- Treat model output as a screening and analytics aid only; do not use it as a diagnostic or employment decision tool.

## Dashboard Inputs Prepared

- Sentiment model results
- Sentiment distribution
- Sentiment by gender
- Sentiment by country
- Treatment by family history
- Treatment by work interference
- Treatment by age group
- Cleaned dataset
- Exported NLP sentiment model
"""

    findings_summary = """# Findings Summary Before Dashboard

The project now follows the internship PDF up to the point where dashboard development should begin. Data preparation, text preprocessing, EDA, TF-IDF feature extraction, sentiment model training, evaluation, model selection, sentiment distribution analysis, at-risk cohort identification, HR recommendations, and model export are prepared.

The next PDF task is to build the Employee Wellness Dashboard using the generated visuals, CSV summaries, cleaned dataset, and exported model.
"""

    (REPORTS_DIR / "sentiment_analysis_report.md").write_text(sentiment_report, encoding="utf-8")
    (REPORTS_DIR / "at_risk_cohort_report.md").write_text(risk_report, encoding="utf-8")
    (REPORTS_DIR / "hr_recommendations.md").write_text(hr_report, encoding="utf-8")
    (REPORTS_DIR / "findings_summary_before_dashboard.md").write_text(findings_summary, encoding="utf-8")


def main() -> None:
    df = pd.read_csv(DATA_DIR / "cleaned_survey.csv")
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[18, 25, 35, 45, 70],
        labels=["18-25", "26-35", "36-45", "46-70"],
        include_lowest=True,
    )

    create_eda_visuals(df)

    comments_df = df[df["comments"].notna()].copy()
    comments_df["clean_comments"] = comments_df["comments"].apply(clean_text)
    comments_df = comments_df[comments_df["clean_comments"].str.len() > 0].copy()
    comments_df["polarity"] = comments_df["comments"].apply(lambda text: TextBlob(str(text)).sentiment.polarity)
    comments_df["sentiment"] = comments_df["comments"].apply(sentiment_label)
    comments_df.to_csv(DATA_DIR / "processed_comments_sentiment.csv", index=False)

    model_info = train_sentiment_models(comments_df)
    joblib.dump(model_info["best_model"], MODELS_DIR / "sentiment_tfidf_pipeline.pkl")

    create_sentiment_visuals(comments_df, model_info)
    build_reports(df, comments_df, model_info)

    print("Completed project artifacts up to dashboard stage.")
    print(f"Best sentiment model: {model_info['best_name']}")
    print(model_info["results"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

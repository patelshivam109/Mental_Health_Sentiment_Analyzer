# Sentiment Analysis Report

## Scope

This report covers the project work up to the dashboard stage from the internship PDF. The source dataset does not contain ground-truth sentiment labels, so comment sentiment labels were generated with TextBlob polarity and then used to train TF-IDF based classifiers.

## Data Used

- Total cleaned survey rows: 1250
- Rows with usable employee comments: 160
- Train rows: 120
- Test rows: 40

## Sentiment Label Distribution

| Sentiment | Count |
| --- | --- |
| Positive | 71 |
| Negative | 50 |
| Neutral | 39 |

## Model Comparison

| Model | Accuracy | Weighted F1 |
| --- | --- | --- |
| Logistic Regression | 0.425 | 0.4197 |
| Tuned Logistic Regression | 0.425 | 0.4072 |
| Naive Bayes | 0.5 | 0.3745 |
| Linear SVM | 0.375 | 0.3619 |

## Selected Model

Best model selected by weighted F1 score: **Logistic Regression**.

## Classification Report

```text
              precision    recall  f1-score   support

    Negative       0.55      0.50      0.52        12
     Neutral       0.11      0.10      0.11        10
    Positive       0.50      0.56      0.53        18

    accuracy                           0.42        40
   macro avg       0.39      0.39      0.38        40
weighted avg       0.42      0.42      0.42        40

```

## Notes

Because the sentiment labels are automatically derived rather than manually annotated, this model should be treated as an internship/project NLP workflow artifact, not a clinical or HR decision system.

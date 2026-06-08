# At-Risk Cohort Identification Report

## Basis

At-risk cohorts were identified using survey signals from the PDF task: treatment seeking, work interference, family history, age group, location, and negative sentiment in free-text comments. The dataset does not include a role/job-title column, so role-based cohort analysis could not be performed exactly.

## Treatment by Family History

| family_history | No | Yes |
| --- | --- | --- |
| No | 492 | 270 |
| Yes | 127 | 361 |

## Treatment by Work Interference

| work_interfere | No | Yes |
| --- | --- | --- |
| Don't know | 258 | 4 |
| Never | 182 | 29 |
| Often | 21 | 119 |
| Rarely | 51 | 122 |
| Sometimes | 107 | 357 |

## Treatment by Age Group

| Age_Group | No | Yes |
| --- | --- | --- |
| 18-25 | 112 | 105 |
| 26-35 | 360 | 341 |
| 36-45 | 124 | 153 |
| 46-70 | 23 | 32 |

## Comment Sentiment by Gender

| Gender | Negative | Neutral | Positive |
| --- | --- | --- | --- |
| Female | 4 | 1 | 1 |
| Male | 46 | 38 | 69 |
| Other | 0 | 0 | 1 |

## Highest Negative-Sentiment Countries Among Comment Rows

| Country | Negative | Neutral | Positive |
| --- | --- | --- | --- |
| Belgium | 1.0 | 0.0 | 0.0 |
| Ireland | 1.0 | 0.0 | 0.0 |
| Portugal | 1.0 | 0.0 | 0.0 |
| South Africa | 1.0 | 0.0 | 0.0 |
| Australia | 0.6 | 0.0 | 0.4 |
| Singapore | 0.5 | 0.0 | 0.5 |
| United States | 0.343 | 0.235 | 0.422 |
| Netherlands | 0.25 | 0.25 | 0.5 |
| United Kingdom | 0.227 | 0.273 | 0.5 |
| Germany | 0.125 | 0.375 | 0.5 |

## Key At-Risk Groups

- Employees reporting that mental health often or sometimes interferes with work.
- Employees with a family history of mental illness.
- Employees in age groups with higher treatment-seeking rates, especially 36-45 and 46-70 in this dataset.
- Employee comment groups with higher negative sentiment rates.

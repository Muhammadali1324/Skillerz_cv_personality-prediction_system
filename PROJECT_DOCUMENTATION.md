# Project Documentation

## Title
**Personality Prediction System Through CV Analysis**

## Problem statement
Recruiters routinely infer “culture fit” from resumes by gut feel. That process is slow, inconsistent, and hard to explain. This project uses machine learning to read a candidate’s CV — word choice, accomplishments, and professional experience — and map the text onto the Big Five personality traits so hiring teams can make more consistent, data-driven shortlists.

## Objectives
1. Accept unstructured CVs (PDF, DOCX, TXT).
2. Extract linguistic and structural features from the text.
3. Predict scores for Openness, Conscientiousness, Extraversion, Agreeableness, and Emotional Stability.
4. Rank the candidate against common role archetypes.
5. Show evidence words and interview prompts so the score is not a black box.

## Tech stack
| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Web | Flask, Jinja2, vanilla JS, Chart.js |
| ML | scikit-learn Gradient Boosting (multi-output) |
| NLP features | Custom LIWC-style lexicons + resume structure |
| Parsing | pypdf, python-docx |
| Storage | SQLite (`instance/candidates.db`) |

## Modules
- `utils/resume_parser.py` — file-to-text
- `utils/feature_extractor.py` — 32 numeric features + evidence terms
- `utils/predictor.py` — model load, calibration, role-fit, insights
- `models/train_model.py` — synthetic CV generator + trainer
- `app.py` — routes: home, analyze, results, dashboard, compare, download, JSON API

## Algorithm (short)
1. Tokenize CV text.
2. Compute lexicon densities for each OCEAN category, plus neuroticism, power, and positive-emotion words.
3. Add structure features: type–token ratio, action-verb density, quantified achievements (`23%`), education level, years of experience, section coverage.
4. Feed the vector to `MultiOutputRegressor(GradientBoostingRegressor)`.
5. Blend model output with lexicon scores for explainability.
6. Compute role-fit as a weighted distance to 12 target profiles.

Hold-out quality of the shipped model (1,600 synthetic CVs, 18% test):

| Trait | MAE | R² |
|---|---|---|
| Openness | 4.72 | 0.87 |
| Conscientiousness | 5.38 | 0.74 |
| Extraversion | 5.89 | 0.84 |
| Agreeableness | 5.93 | 0.76 |
| Emotional stability | 4.99 | 0.76 |

## Screens
1. **Home** — product story and pipeline overview
2. **Analyze** — drag-and-drop upload or paste
3. **Results** — radar chart, trait cards, role fit, recruiter brief
4. **Recruiter desk** — slate table + average personality mix
5. **Compare** — overlay 2–3 candidates

## How to run
See `START_HERE.txt` and `README.md`.

```
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and upload a file from `data/sample_cvs/`.

## Ethics
Personality inferred from a CV is a **weak proxy**. Language is shaped by culture, coaching, and template CVs. The UI states this clearly. The system must not auto-reject candidates.

## Future work
- Fine-tune a transformer on consented, labelled CVs
- Multilingual support (Urdu / Arabic / Chinese resumes)
- Bias audits by gender and first-language markers
- ATS integration (Greenhouse / Lever webhooks)

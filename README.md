# Personality Prediction System Through CV Analysis

**PersonaCV** is a complete Python Flask + machine-learning project that reads a candidate’s resume and predicts **Big Five (OCEAN)** personality traits:

| Trait | What the model looks for in a CV |
|---|---|
| **Openness** | Research, design, novelty, abstract / creative language |
| **Conscientiousness** | Delivery, process, certifications, quantified achievements |
| **Extraversion** | Leadership, presenting, events, stakeholder language |
| **Agreeableness** | Mentoring, volunteering, empathy, inclusive wording |
| **Emotional stability** | Composure, ownership, crisis delivery (vs. stress framing) |

Recruiters get role-fit rankings, language evidence, culture notes, and interview prompts so hiring stays data-informed — not gut-feel only.

> This is a **decision-support** tool inferred from CV language. It is **not** a clinical psychological test and must never be the sole basis for a hiring decision.

---

## Features

- Upload **PDF / DOCX / TXT** or paste resume text
- Linguistic feature extractor (32 features: lexicons, TTR, action verbs, numbers, education, structure)
- **Multi-output Gradient Boosting** model (`scikit-learn`)
- Radar chart + per-trait evidence words
- 12 role-fit archetypes (SWE, data science, sales, PM, UX, HR, …)
- Recruiter desk, candidate compare, downloadable text report
- JSON API at `POST /api/analyze`
- Four realistic sample CVs in `data/sample_cvs/`

---

## Project layout

```
personality-prediction-system/
├── app.py                      # Flask application
├── requirements.txt
├── models/
│   ├── train_model.py          # Retrain on synthetic CVs
│   └── personality_model.joblib
├── utils/
│   ├── resume_parser.py        # PDF / DOCX / TXT
│   ├── feature_extractor.py    # Linguistic features + evidence
│   └── predictor.py            # Scores, role-fit, insights
├── templates/  static/
├── data/sample_cvs/            # Demo resumes
├── uploads/                    # Runtime uploads (gitignored)
└── instance/candidates.db      # Created on first run
```

---

## Quick start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) retrain the model
python models/train_model.py

# 4. Run the app
python app.py
```

Open **http://127.0.0.1:5000**

Try the sample files in `data/sample_cvs/` first:

- `amina_qureshi_data_scientist.txt` — high openness / research
- `luis_ortega_sales.txt` — high extraversion
- `priya_nair_project_manager.txt` — high conscientiousness
- `chen_wei_ux_designer.txt` — high openness + agreeableness

---

## How the ML works

1. **Parse** the CV to plain text (`pypdf` / `python-docx`).
2. **Extract features** — LIWC-style category densities plus structural signals (word count, type–token ratio, bullet density, `%` achievements, education level, years of experience, action-verb density).
3. **Predict** five continuous scores with a `MultiOutputRegressor(GradientBoostingRegressor)` trained on **1,600 synthetic CVs**. Each synthetic resume is generated from an OCEAN archetype so the language–trait mapping follows published correlates (no private HR data required).
4. **Calibrate & explain** — blend model output with lexicon evidence so every score can point at words that actually appeared in the file.
5. **Role fit** — Euclidean distance (weighted) against 12 hiring profiles, converted to a 0–100 similarity.

Retrain anytime:

```bash
python models/train_model.py
```

If the `.joblib` file is missing, the app automatically falls back to a transparent heuristic so demos never crash.

### JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -F "cv=@data/sample_cvs/amina_qureshi_data_scientist.txt"
```

Or JSON:

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"Amina\",\"text\":\"...full CV text...\"}"
```

---

## Academic / project notes

This repository is designed as a **final-year / portfolio project**:

- Problem statement matches *Personality Prediction System Through CV Analysis*
- Stack is **Python + Flask + scikit-learn** (classic, easy to viva)
- You can discuss feature engineering, multi-output regression, MAE / R² from `train_model.py`, and ethical limits of inferring personality from CVs

Suggested viva talking points:

1. Why Big Five rather than MBTI (trait theory is continuous and better studied in I/O psychology).
2. Why synthetic training data (privacy + no labelled corporate CVs).
3. Why explainability (lexicon evidence) matters more than a 2-point MAE gain.
4. Fairness: language on a CV is culturally biased; always pair with work samples.

---

## License

Provided for education and portfolio use. Do not deploy as an automated rejector.

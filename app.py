"""
Personality Prediction System Through CV Analysis
Flask + scikit-learn application for recruiter decision support.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from utils.predictor import PersonalityPredictor, TRAIT_META
from utils.resume_parser import allowed_file, extract_text

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
DB_PATH = ROOT / "instance" / "candidates.db"
ALLOWED = {"pdf", "docx", "txt", "doc"}

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-change-me-in-production"),
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,
    UPLOAD_FOLDER=str(UPLOAD_DIR),
)

UPLOAD_DIR.mkdir(exist_ok=True)
(ROOT / "instance").mkdir(exist_ok=True)

predictor = PersonalityPredictor()


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            public_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            target_role TEXT,
            filename TEXT,
            text_preview TEXT,
            word_count INTEGER,
            scores_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    db.commit()
    db.close()


init_db()


def row_to_candidate(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["scores"] = json.loads(item["scores_json"])
    item["result"] = json.loads(item["result_json"])
    return item


@app.context_processor
def inject_globals():
    return {
        "trait_meta": TRAIT_META,
        "trait_order": list(TRAIT_META.keys()),
        "year": datetime.utcnow().year,
    }


@app.route("/")
def index():
    db = get_db()
    count = db.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()["n"]
    latest = db.execute(
        "SELECT id, public_id, name, target_role, scores_json, created_at FROM candidates ORDER BY id DESC LIMIT 4"
    ).fetchall()
    cards = []
    for row in latest:
        cards.append(
            {
                "public_id": row["public_id"],
                "name": row["name"],
                "target_role": row["target_role"],
                "scores": json.loads(row["scores_json"]),
                "created_at": row["created_at"],
            }
        )
    return render_template("index.html", analyzed=count, latest=cards)


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        file = request.files.get("cv")
        name = (request.form.get("name") or "").strip()
        target_role = (request.form.get("target_role") or "").strip()
        pasted = (request.form.get("pasted_text") or "").strip()

        text = ""
        filename = ""
        if file and file.filename:
            if not allowed_file(file.filename):
                flash("Please upload a PDF, DOCX, or TXT resume.", "error")
                return redirect(url_for("analyze"))
            filename = secure_filename(file.filename)
            stored = f"{uuid.uuid4().hex}_{filename}"
            path = UPLOAD_DIR / stored
            file.save(path)
            try:
                text = extract_text(path, filename)
            except Exception as exc:
                flash(f"Could not read that file: {exc}", "error")
                return redirect(url_for("analyze"))
        elif pasted:
            text = pasted
            filename = "pasted.txt"
        else:
            flash("Upload a CV or paste the resume text to continue.", "error")
            return redirect(url_for("analyze"))

        if len(text.split()) < 40:
            flash("That CV is too short to score reliably. Add more experience detail (40+ words).", "error")
            return redirect(url_for("analyze"))

        result = predictor.predict(text)
        if not name:
            # First non-empty line is often the candidate name
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "Unnamed candidate")
            name = first[:80]

        public_id = uuid.uuid4().hex[:12]
        db = get_db()
        db.execute(
            """
            INSERT INTO candidates
                (public_id, name, target_role, filename, text_preview, word_count, scores_json, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                public_id,
                name,
                target_role,
                filename,
                text[:800],
                result["summary"]["word_count"],
                json.dumps(result["scores"]),
                json.dumps(result),
                datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            ),
        )
        db.commit()
        return redirect(url_for("results", public_id=public_id))

    return render_template("analyze.html")


@app.route("/results/<public_id>")
def results(public_id: str):
    row = get_db().execute("SELECT * FROM candidates WHERE public_id = ?", (public_id,)).fetchone()
    if not row:
        abort(404)
    candidate = row_to_candidate(row)
    return render_template("results.html", c=candidate)


@app.route("/dashboard")
def dashboard():
    rows = get_db().execute("SELECT * FROM candidates ORDER BY id DESC").fetchall()
    candidates = [row_to_candidate(r) for r in rows]
    averages = {k: 0 for k in TRAIT_META}
    if candidates:
        for k in averages:
            averages[k] = round(sum(c["scores"][k] for c in candidates) / len(candidates))
    return render_template("dashboard.html", candidates=candidates, averages=averages)


@app.route("/compare")
def compare():
    ids = request.args.getlist("id")
    db = get_db()
    all_rows = db.execute(
        "SELECT public_id, name, target_role, scores_json, created_at FROM candidates ORDER BY id DESC"
    ).fetchall()
    catalog = [
        {
            "public_id": r["public_id"],
            "name": r["name"],
            "target_role": r["target_role"],
            "scores": json.loads(r["scores_json"]),
            "created_at": r["created_at"],
        }
        for r in all_rows
    ]
    selected = [c for c in catalog if c["public_id"] in ids][:3]
    # If ids were passed but some missing, just use what we have
    return render_template("compare.html", catalog=catalog, selected=selected)


@app.route("/download/<public_id>")
def download_report(public_id: str):
    row = get_db().execute("SELECT * FROM candidates WHERE public_id = ?", (public_id,)).fetchone()
    if not row:
        abort(404)
    c = row_to_candidate(row)
    r = c["result"]
    lines = [
        "PERSONALITY PREDICTION SYSTEM — CANDIDATE REPORT",
        "=" * 56,
        f"Candidate : {c['name']}",
        f"Target role: {c['target_role'] or 'Not specified'}",
        f"Analyzed  : {c['created_at']} UTC",
        f"Source CV : {c['filename'] or 'pasted text'}",
        f"Word count: {c['word_count']}",
        "",
        "BIG FIVE SCORES (0–100)",
        "-" * 56,
    ]
    for t in r["traits"]:
        bar = "█" * (t["score"] // 5) + "░" * (20 - t["score"] // 5)
        lines.append(f"{t['label']:<22} {t['score']:>3}  {bar}  {t['level']}")
        lines.append(f"  {t['description']}")
        if t["signals"]:
            lines.append(f"  Language signals: {', '.join(t['signals'][:6])}")
        lines.append("")
    lines += ["ROLE FIT", "-" * 56]
    for role in r["role_fit"][:6]:
        lines.append(f"  {role['role']:<28} {role['score']:>3}")
    lines += ["", "RECRUITER NOTES", "-" * 56, r["insights"]["headline"], ""]
    lines.append("Culture fit")
    for item in r["insights"]["culture"]:
        lines.append(f"  • {item}")
    if r["insights"]["watchouts"]:
        lines.append("Watch-outs")
        for item in r["insights"]["watchouts"]:
            lines.append(f"  • {item}")
    lines.append("Interview prompts")
    for item in r["insights"]["interview_prompts"]:
        lines.append(f"  • {item}")
    lines += [
        "",
        "DISCLAIMER",
        "This report is a decision-support signal inferred from CV language.",
        "It is not a clinical psychological assessment and must not be the",
        "sole basis for a hiring decision. Combine with structured interviews,",
        "work samples, and legally compliant evaluation criteria.",
        "",
        "Generated by Personality Prediction System Through CV Analysis",
    ]
    report = "\n".join(lines)
    out = UPLOAD_DIR / f"report_{public_id}.txt"
    out.write_text(report, encoding="utf-8")
    safe = secure_filename(c["name"]) or "candidate"
    return send_file(out, as_attachment=True, download_name=f"{safe}_personality_report.txt")


@app.route("/delete/<public_id>", methods=["POST"])
def delete_candidate(public_id: str):
    get_db().execute("DELETE FROM candidates WHERE public_id = ?", (public_id,))
    get_db().commit()
    flash("Candidate removed from the workspace.", "ok")
    return redirect(url_for("dashboard"))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """JSON API: multipart file `cv` or JSON {text, name}."""
    text = ""
    name = ""
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        name = (payload.get("name") or "").strip()
    else:
        name = (request.form.get("name") or "").strip()
        if request.files.get("cv"):
            f = request.files["cv"]
            if not allowed_file(f.filename):
                return {"error": "Unsupported file type"}, 400
            text = extract_text(f, f.filename)
        else:
            text = (request.form.get("text") or "").strip()

    if len(text.split()) < 40:
        return {"error": "Provide at least 40 words of CV text."}, 400

    result = predictor.predict(text)
    return {
        "name": name or None,
        "scores": result["scores"],
        "traits": result["traits"],
        "role_fit": result["role_fit"][:5],
        "insights": result["insights"],
        "summary": result["summary"],
        "model_used": result["model_used"],
    }


@app.errorhandler(404)
def not_found(_e):
    return render_template("404.html"), 404


@app.errorhandler(413)
def too_large(_e):
    flash("File is larger than 8 MB.", "error")
    return redirect(url_for("analyze"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

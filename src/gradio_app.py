import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile
import traceback
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

from resume_parser import extract_text_from_pdf, extract_skills_from_text
from assessment_engine import generate_assessment, grade_assessment
from database import get_or_create_user, save_verified_skills, save_recommendation, init_db, get_user_dashboard
from report_generator import generate_pdf_report
from ml_engine import load_or_train_models, build_knn_matches_html, build_forecast_html

init_db()

# Train / load ML models at startup
print("🤖 Loading ML models...")
_knn_model, _skill_forecaster = load_or_train_models()

DATA_DIR     = os.path.join(BASE_DIR, '..', 'data')
matrix_df    = pd.read_csv(os.path.join(DATA_DIR, "job_skill_matrix.csv"))
job_titles   = matrix_df['Job Title']
skill_matrix = matrix_df.drop(columns=['Job Title'])
skills_list  = skill_matrix.columns.tolist()

# Load skill frequency for trend analysis
freq_df      = pd.read_csv(os.path.join(DATA_DIR, "skill_frequency.csv"))
freq_df['Skill'] = freq_df['Skill'].str.lower().str.strip()
max_count    = freq_df['Count'].max()

MAX_QUESTIONS = 25
MAX_CERTS     = 5

custom_css = """
.gradio-container { max-width: 1050px !important; margin: auto; font-family: 'Inter', sans-serif; }
.app-header { text-align:center; padding:30px 0 20px; margin-bottom:20px; border-bottom:1px solid #1e293b; }
.app-header h1 { font-weight:800; font-size:2.6em; margin-bottom:6px;
    background:linear-gradient(135deg,#10b981,#3b82f6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.app-header p { font-size:1.05em; color:#94a3b8; }
.btn-primary { background:linear-gradient(to right,#059669,#10b981) !important;
    border:none !important; color:white !important; font-weight:700 !important;
    font-size:1em !important; border-radius:10px !important; padding:14px !important; }
.btn-add    { background:rgba(59,130,246,0.15) !important; border:1px solid #3b82f6 !important;
    color:#60a5fa !important; font-weight:600 !important; border-radius:8px !important; }
.btn-remove { background:rgba(239,68,68,0.1) !important; border:1px solid #ef4444 !important;
    color:#f87171 !important; font-weight:600 !important; border-radius:8px !important; }
.skill-chip { background:rgba(16,185,129,0.12); color:#10b981;
    border:1px solid rgba(16,185,129,0.35); padding:5px 12px; border-radius:20px;
    display:inline-block; margin:3px; font-weight:600; font-size:0.85em; }
.cert-chip  { background:rgba(59,130,246,0.12); color:#60a5fa;
    border:1px solid rgba(59,130,246,0.35); padding:5px 12px; border-radius:20px;
    display:inline-block; margin:3px; font-weight:600; font-size:0.85em; }
.cert-card  { border:1px solid #1e293b !important; border-left:4px solid #3b82f6 !important;
    border-radius:10px !important; padding:16px !important; margin-bottom:12px !important;
    background:#0f172a !important; }
.question-box { border:1px solid #1e293b !important; border-left:4px solid #10b981 !important;
    border-radius:10px !important; padding:18px 20px !important; margin-bottom:14px !important;
    background:#0f172a !important; }
"""

# ═══════════════════════════════════════════════════════════════════
# HELPER: Answer review HTML
# ═══════════════════════════════════════════════════════════════════
def build_review_html(quiz_state, user_answers):
    if not quiz_state:
        return ""
    html = "<div style='padding:10px 0;'>"
    correct_count = 0
    total = len(quiz_state)
    for i, item in enumerate(quiz_state):
        question    = item.get("question", "")
        options     = item.get("options", [])
        correct     = str(item.get("correct_answer", "")).strip().upper()
        skill       = str(item.get("skill", "")).upper()
        user_raw    = user_answers[i] if i < len(user_answers) and user_answers[i] else None
        user_letter = str(user_raw).strip().upper()[0] if user_raw else None
        is_correct  = user_letter == correct
        if is_correct: correct_count += 1
        cb = "#10b981" if is_correct else "#ef4444"
        bb = "#10b981" if is_correct else "#ef4444"
        bt = "✓ Correct" if is_correct else "✗ Incorrect"
        bg_card = "rgba(16,185,129,0.05)" if is_correct else "rgba(239,68,68,0.05)"
        html += f"""
        <div style='margin-bottom:16px;border:1px solid {cb};border-left:5px solid {cb};
                    border-radius:10px;padding:18px;background:{bg_card};'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;'>
                <p style='margin:0;font-weight:700;color:#e2e8f0;font-size:0.95em;flex:1;'>
                    <span style='color:#64748b;font-size:0.85em;'>[{skill}]</span><br/>Q{i+1}. {question}
                </p>
                <span style='background:{bb};color:white;padding:4px 12px;border-radius:20px;
                             font-size:0.8em;font-weight:700;margin-left:12px;white-space:nowrap;'>{bt}</span>
            </div>
            <div style='display:flex;flex-direction:column;gap:8px;'>"""
        for opt in options:
            ol = str(opt).strip().upper()[0] if opt else ""
            ico = isc = isu = False
            isc = ol == correct
            isu = ol == user_letter
            if isc and isu:   bg,fc,bc,ic,fw = "linear-gradient(to right,#065f46,#10b981)","white","#10b981","✓ ","700"
            elif isc:         bg,fc,bc,ic,fw = "rgba(16,185,129,0.12)","#10b981","#10b981","✓ ","600"
            elif isu:         bg,fc,bc,ic,fw = "linear-gradient(to right,#7f1d1d,#ef4444)","white","#ef4444","✗ ","700"
            else:             bg,fc,bc,ic,fw = "rgba(30,41,59,0.6)","#94a3b8","#334155","","400"
            html += f"""<div style='padding:10px 16px;border-radius:8px;border:1px solid {bc};
                            background:{bg};color:{fc};font-weight:{fw};font-size:0.9em;'>{ic}{opt}</div>"""
        html += "</div></div>"
    pct = round((correct_count / total) * 100) if total else 0
    if pct >= 70:   sc,se,sm = "#10b981","🏆","Great job!"
    elif pct >= 40: sc,se,sm = "#f59e0b","📈","Keep practising!"
    else:           sc,se,sm = "#ef4444","📚","More study needed!"
    summary = f"""<div style='margin-bottom:20px;padding:20px;border-radius:12px;
        background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid {sc};text-align:center;'>
        <div style='font-size:2.5em;'>{se}</div>
        <p style='color:{sc};font-size:1.5em;font-weight:800;margin:6px 0;'>
            {correct_count} / {total} Correct — {pct}%</p>
        <p style='color:#94a3b8;margin:0;'>{sm} Scroll down to see your career analysis.</p>
    </div>"""
    return summary + html + "</div>"


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — FEATURE 1: Top 3 Job Matches HTML
# ═══════════════════════════════════════════════════════════════════
def build_top3_jobs_html(similarity_scores, user_vector):
    scores_flat = similarity_scores[0]
    top3_indices = np.argsort(scores_flat)[::-1][:3]

    medals = ["🥇", "🥈", "🥉"]
    colors = ["#10b981", "#3b82f6", "#8b5cf6"]
    labels = ["Best Match", "Strong Match", "Good Match"]

    html = """
    <div style='margin-bottom:8px;'>
        <h3 style='color:#e2e8f0;font-size:1.1em;margin:0 0 16px;'>💼 Top 3 Matching Career Paths</h3>"""

    for rank, idx in enumerate(top3_indices):
        job_name  = job_titles.iloc[idx]
        raw_score = float(scores_flat[idx])
        pct       = min(round(raw_score * 100), 99) if raw_score < 1 else 99

        # missing skills for this job
        job_req = skill_matrix.iloc[idx]
        missing = [s for i, s in enumerate(skills_list)
                   if job_req.iloc[i] == 1 and user_vector[i] < 3][:4]
        missing_html = " ".join([f"<span style='background:rgba(239,68,68,0.12);color:#f87171;"
                                 f"border:1px solid rgba(239,68,68,0.3);padding:2px 8px;"
                                 f"border-radius:12px;font-size:0.75em;margin:2px;display:inline-block;'>"
                                 f"+{s.title()}</span>" for s in missing])

        c = colors[rank]
        html += f"""
        <div style='margin-bottom:14px;padding:16px;border-radius:10px;
                    border:1px solid {c};background:rgba(0,0,0,0.2);'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                <div>
                    <span style='font-size:1.3em;margin-right:8px;'>{medals[rank]}</span>
                    <span style='color:#e2e8f0;font-weight:700;font-size:0.95em;'>{job_name}</span>
                    <span style='color:#64748b;font-size:0.78em;margin-left:8px;'>{labels[rank]}</span>
                </div>
                <span style='color:{c};font-weight:800;font-size:1.1em;'>{pct}%</span>
            </div>
            <div style='width:100%;background:#1e293b;border-radius:6px;margin-bottom:10px;'>
                <div style='width:{pct}%;background:{c};height:8px;border-radius:6px;'></div>
            </div>
            <div style='font-size:0.78em;color:#64748b;'>
                Skills to bridge gap: {missing_html if missing else "<span style='color:#10b981;'>✅ You meet all requirements!</span>"}
            </div>
        </div>"""

    html += "</div>"
    return html


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — FEATURE 2: Job Trend Analysis HTML
# ═══════════════════════════════════════════════════════════════════
def build_trend_html(user_skills):
    user_skills_lower = [s.lower().strip() for s in user_skills]

    # Trending = top 15 by count
    trending  = freq_df.head(15)
    # Niche     = count between 5 and 15
    niche     = freq_df[(freq_df['Count'] >= 5) & (freq_df['Count'] <= 15)].head(8)

    def trend_bar(count, max_c=82):
        pct = round((count / max_c) * 100)
        if pct >= 70:   color = "#10b981"
        elif pct >= 40: color = "#f59e0b"
        else:           color = "#3b82f6"
        return pct, color

    html = """<div>
    <h3 style='color:#e2e8f0;font-size:1.1em;margin:0 0 14px;'>📈 Job Market Trend Analysis</h3>
    <p style='color:#64748b;font-size:0.82em;margin-bottom:16px;'>
        Based on 690+ skills across 493 tech job roles. Green highlight = you already have it!
    </p>"""

    # Hot skills
    html += """<div style='margin-bottom:20px;'>
        <p style='color:#10b981;font-weight:700;font-size:0.85em;margin-bottom:10px;'>
            🔥 Most In-Demand Skills Right Now
        </p>"""
    for _, row in trending.iterrows():
        skill = row['Skill']
        count = row['Count']
        pct, color = trend_bar(count)
        have_it = skill in user_skills_lower
        bg  = "rgba(16,185,129,0.08)" if have_it else "rgba(15,23,42,0.8)"
        tag = "<span style='color:#10b981;font-size:0.7em;font-weight:700;'>✓ YOU HAVE IT</span>" if have_it else ""
        html += f"""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;
                    padding:8px 12px;border-radius:8px;background:{bg};border:1px solid #1e293b;'>
            <span style='color:#cbd5e1;font-size:0.85em;min-width:160px;'>{skill.title()}</span>
            <div style='flex:1;background:#1e293b;border-radius:4px;height:6px;'>
                <div style='width:{pct}%;background:{color};height:6px;border-radius:4px;'></div>
            </div>
            <span style='color:{color};font-size:0.8em;font-weight:700;min-width:40px;text-align:right;'>
                {count}</span>
            <span style='min-width:90px;'>{tag}</span>
        </div>"""
    html += "</div>"

    # Niche / emerging skills
    html += """<div>
        <p style='color:#8b5cf6;font-weight:700;font-size:0.85em;margin-bottom:10px;'>
            💎 Niche & Emerging Skills (Low competition, high value)
        </p>"""
    for _, row in niche.iterrows():
        skill = row['Skill']
        count = row['Count']
        have_it = skill in user_skills_lower
        tag = "✓" if have_it else ""
        color = "#10b981" if have_it else "#8b5cf6"
        html += f"""
        <span style='display:inline-block;margin:3px;padding:5px 12px;border-radius:20px;
                     background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.3);
                     color:{color};font-size:0.82em;font-weight:600;'>{tag} {skill.title()} ({count})</span>"""
    html += "</div></div>"
    return html


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — FEATURE 3: Learning Path Generator HTML
# ═══════════════════════════════════════════════════════════════════
LEARNING_PATHS = {
    "python":           {"weeks": 6,  "resources": ["Python.org docs", "Automate the Boring Stuff (free)", "freeCodeCamp Python course"], "prereq": None},
    "machine learning": {"weeks": 10, "resources": ["Andrew Ng ML Course (Coursera)", "Scikit-learn docs", "Kaggle Learn"], "prereq": "python"},
    "deep learning":    {"weeks": 8,  "resources": ["fast.ai (free)", "deeplearning.ai", "PyTorch tutorials"], "prereq": "machine learning"},
    "sql":              {"weeks": 4,  "resources": ["SQLZoo (free)", "Mode Analytics SQL Tutorial", "LeetCode SQL"], "prereq": None},
    "docker":           {"weeks": 3,  "resources": ["Docker official docs", "Play with Docker (free lab)", "TechWorld with Nana (YouTube)"], "prereq": "linux"},
    "kubernetes":       {"weeks": 5,  "resources": ["Kubernetes.io docs", "KodeKloud (free tier)", "CKAD prep guide"], "prereq": "docker"},
    "aws":              {"weeks": 6,  "resources": ["AWS Free Tier + docs", "A Cloud Guru", "AWS Skill Builder (free)"], "prereq": None},
    "azure":            {"weeks": 6,  "resources": ["Microsoft Learn (free)", "AZ-900 study guide", "Azure free account"], "prereq": None},
    "javascript":       {"weeks": 6,  "resources": ["javascript.info (free)", "The Odin Project (free)", "freeCodeCamp JS"], "prereq": None},
    "react":            {"weeks": 5,  "resources": ["React official docs", "Scrimba React course", "Full Stack Open (free)"], "prereq": "javascript"},
    "linux":            {"weeks": 3,  "resources": ["Linux Journey (free)", "OverTheWire Bandit", "The Linux Command Line (free PDF)"], "prereq": None},
    "git":              {"weeks": 2,  "resources": ["Pro Git book (free)", "Learn Git Branching (interactive)", "GitHub Skills"], "prereq": None},
    "ci/cd":            {"weeks": 4,  "resources": ["GitHub Actions docs", "Jenkins tutorials", "GitLab CI/CD docs"], "prereq": "git"},
    "cloud computing":  {"weeks": 5,  "resources": ["AWS/Azure/GCP free tiers", "Cloud Guru", "Google Cloud Skills Boost"], "prereq": None},
    "data analysis":    {"weeks": 5,  "resources": ["Kaggle Python (free)", "Pandas docs", "DataCamp free tier"], "prereq": "python"},
    "security":         {"weeks": 8,  "resources": ["TryHackMe (free rooms)", "CompTIA Security+ guide", "OWASP Top 10 docs"], "prereq": "networking"},
    "networking":       {"weeks": 4,  "resources": ["Professor Messer (free)", "Cisco NetAcad", "NetworkChuck (YouTube)"], "prereq": None},
    "java":             {"weeks": 8,  "resources": ["MOOC.fi Java course (free)", "Codecademy Java", "Baeldung tutorials"], "prereq": None},
    "terraform":        {"weeks": 4,  "resources": ["HashiCorp Learn (free)", "Terraform docs", "TechWorld with Nana"], "prereq": "aws"},
    "ansible":          {"weeks": 3,  "resources": ["Ansible docs", "Red Hat free training", "Jeff Geerling tutorials"], "prereq": "linux"},
}

def build_learning_path_html(missing_skills, best_job):
    if not missing_skills:
        return """<div style='padding:20px;border-radius:10px;border:1px solid #10b981;
            background:rgba(16,185,129,0.07);text-align:center;'>
            <div style='font-size:2em;'>🎉</div>
            <p style='color:#10b981;font-weight:700;'>You already meet all skill requirements!</p>
            <p style='color:#64748b;font-size:0.85em;'>Consider applying for <b>{best_job}</b> now.</p>
        </div>"""

    # Filter to skills we have paths for, then fall back to generic
    priority = [s for s in missing_skills[:8] if s.lower() in LEARNING_PATHS]
    generic  = [s for s in missing_skills[:8] if s.lower() not in LEARNING_PATHS]

    total_weeks = sum(LEARNING_PATHS.get(s.lower(), {}).get("weeks", 3) for s in priority)

    html = f"""<div>
    <div style='padding:14px 18px;border-radius:10px;background:linear-gradient(135deg,#0f172a,#1e293b);
                border:1px solid #3b82f6;margin-bottom:20px;'>
        <p style='color:#60a5fa;font-weight:700;font-size:0.9em;margin:0 0 4px;'>
            🗓️ Estimated Learning Timeline for <b>{best_job}</b>
        </p>
        <p style='color:#e2e8f0;font-size:1.4em;font-weight:800;margin:0;'>
            ~{total_weeks} weeks
        </p>
        <p style='color:#64748b;font-size:0.78em;margin:4px 0 0;'>
            Studying ~1-2 hours/day · {len(priority)} core skills to master
        </p>
    </div>"""

    for i, skill in enumerate(priority):
        info  = LEARNING_PATHS.get(skill.lower(), {})
        weeks = info.get("weeks", 3)
        resources = info.get("resources", ["Search on YouTube", "Check official docs"])
        prereq    = info.get("prereq")

        resources_html = "".join([
            f"<div style='padding:6px 10px;border-radius:6px;background:#1e293b;"
            f"color:#94a3b8;font-size:0.8em;margin-bottom:4px;'>📎 {r}</div>"
            for r in resources
        ])
        prereq_html = (f"<span style='color:#f59e0b;font-size:0.75em;'>⚠️ Learn <b>{prereq.title()}</b> first</span>"
                       if prereq else "")

        html += f"""
        <div style='margin-bottom:14px;border:1px solid #1e293b;border-left:4px solid #3b82f6;
                    border-radius:10px;padding:16px;background:#0f172a;'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                <div>
                    <span style='background:#3b82f6;color:white;padding:2px 10px;border-radius:12px;
                                 font-size:0.75em;font-weight:700;margin-right:8px;'>STEP {i+1}</span>
                    <span style='color:#e2e8f0;font-weight:700;font-size:0.95em;'>{skill.title()}</span>
                </div>
                <span style='color:#3b82f6;font-size:0.82em;font-weight:700;'>~{weeks} weeks</span>
            </div>
            {prereq_html}
            <div style='margin-top:8px;'>{resources_html}</div>
        </div>"""

    if generic:
        chips = " ".join([f"<span style='display:inline-block;margin:3px;padding:4px 10px;border-radius:14px;"
                          f"background:rgba(100,116,139,0.12);border:1px solid #334155;"
                          f"color:#94a3b8;font-size:0.8em;'>{s.title()}</span>" for s in generic])
        html += f"""<div style='padding:14px;border-radius:8px;background:#0f172a;border:1px solid #1e293b;'>
            <p style='color:#64748b;font-size:0.82em;margin:0 0 8px;'>
                📚 Also explore (search on Udemy/YouTube/Coursera):
            </p>{chips}</div>"""

    html += "</div>"
    return html


# ═══════════════════════════════════════════════════════════════════
# Cert card management
# ═══════════════════════════════════════════════════════════════════
def add_cert(count):
    count = min(count + 1, MAX_CERTS)
    return [count] + [gr.update(visible=(i < count)) for i in range(MAX_CERTS)]

def remove_cert(count):
    count = max(count - 1, 1)
    return [count] + [gr.update(visible=(i < count)) for i in range(MAX_CERTS)]


# ═══════════════════════════════════════════════════════════════════
# Process & Quiz
# ═══════════════════════════════════════════════════════════════════
def process_and_quiz(email, pdf_path, api_key, *args, progress=gr.Progress()):
    cert_names  = list(args[0:MAX_CERTS])
    cert_skills = list(args[MAX_CERTS:MAX_CERTS*2])
    cert_files  = list(args[MAX_CERTS*2:MAX_CERTS*3])
    print("\n--- STARTING RESUME PARSE ---")

    def status_html(step, msg, pct):
        return f"""<div style='padding:24px;border-radius:12px;border:1px solid #1e293b;
                    background:#0f172a;text-align:center;'>
            <div style='font-size:2.2em;margin-bottom:8px;'>⏳</div>
            <p style='color:#10b981;font-size:1.05em;font-weight:700;margin-bottom:4px;'>Step {step} of 5</p>
            <p style='color:#cbd5e1;margin-bottom:14px;font-size:0.95em;'>{msg}</p>
            <div style='width:100%;background:#1e293b;border-radius:6px;'>
                <div style='width:{pct}%;background:linear-gradient(to right,#059669,#10b981);
                            height:10px;border-radius:6px;'></div>
            </div>
            <p style='color:#475569;font-size:0.82em;margin-top:8px;'>{pct}% complete</p>
        </div>"""

    def yield_error(msg):
        yield [gr.update(value=f"<div style='padding:16px;border-radius:10px;border:1px solid #ef4444;background:rgba(239,68,68,0.08);'>{msg}</div>", visible=True),
               None, []] + [gr.update(visible=False)] * MAX_QUESTIONS + [gr.update(selected="upload_tab")]

    try:
        progress(0.1, desc="Validating inputs...")
        yield [status_html(1, "Validating your inputs...", 10), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]

        if not email or "@" not in email:
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ Please enter a valid email address.</h3>"); return
        if not api_key:
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ API Key is missing.</h3>"); return
        if not pdf_path:
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ Please upload a PDF resume.</h3>"); return

        progress(0.2, desc="Authenticating user...")
        yield [status_html(2, "Authenticating user via Database...", 25), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        user_id = get_or_create_user(email)

        progress(0.4, desc="Reading PDF...")
        yield [status_html(3, "Reading your PDF resume...", 45), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        text = extract_text_from_pdf(pdf_path)
        for cf in cert_files:
            if cf:
                try: text += " " + extract_text_from_pdf(cf)
                except: pass

        progress(0.55, desc="Extracting skills...")
        yield [status_html(4, "Extracting technical skills...", 60), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        skills = extract_skills_from_text(text)
        for cs in cert_skills:
            if cs and cs.strip():
                for e in [s.strip() for s in cs.replace(",", "\n").split("\n") if s.strip()]:
                    if e not in skills: skills.append(e)

        if not skills:
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ No technical skills found.</h3>"); return

        certs_added = [(cert_names[i] or f"Certificate {i+1}")
                       for i in range(MAX_CERTS)
                       if (cert_names[i] and cert_names[i].strip()) or cert_files[i]
                       or (cert_skills[i] and cert_skills[i].strip())]

        progress(0.7, desc="🤖 Generating assessment...")
        yield [status_html(5, "🤖 AI is crafting your quiz questions...", 75), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        quiz = generate_assessment(skills, api_key)

        if not quiz or not isinstance(quiz, list):
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ AI Generation Failed. Check your API Key.</h3>"); return

        progress(0.95, desc="Building UI...")
        skills_html = " ".join([f"<span class='skill-chip'>{str(s).title()}</span>" for s in skills])
        certs_html  = ""
        if certs_added:
            chips = " ".join([f"<span class='cert-chip'>🎓 {c}</span>" for c in certs_added])
            certs_html = f"<div style='margin-top:12px;padding-top:12px;border-top:1px solid #1e293b;'><p style='color:#60a5fa;font-weight:600;font-size:0.85em;margin-bottom:6px;'>📜 {len(certs_added)} Certificate(s) Added</p>{chips}</div>"

        status_display = f"""<div style='padding:18px;border-radius:12px;border:1px solid #10b981;background:rgba(16,185,129,0.07);'>
            <h3 style='color:#10b981;margin-bottom:10px;font-size:1em;'>✅ Profile Built Successfully</h3>
            <p style='color:#64748b;font-size:0.82em;margin-bottom:8px;'>
                {len(skills)} skills extracted · {len(certs_added)} certificate(s) added
            </p><div>{skills_html}</div>{certs_html}</div>"""

        updates = []
        for i in range(MAX_QUESTIONS):
            if i < len(quiz):
                q = quiz[i]
                raw_opts = q.get('options', ["A) 1", "B) 2", "C) 3", "D) 4"])
                if isinstance(raw_opts, dict):   opts = list(raw_opts.values())
                elif isinstance(raw_opts, list): opts = raw_opts
                else:                            opts = [str(raw_opts)]
                opts = list(dict.fromkeys([str(o) for o in opts]))
                if len(opts) < 2: opts = ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"]
                skill_tag = str(q.get('skill', 'SKILL')).upper()
                updates.append(gr.update(label=f"Q{i+1}. {str(q.get('question',''))}  ·  [{skill_tag}]",
                                         choices=opts, visible=True, value=None))
            else:
                updates.append(gr.update(visible=False))

        progress(1.0, desc="✅ Done!")
        print("7. Success!")
        yield [gr.update(value=status_display, visible=True), user_id, quiz] + updates + [gr.update(selected="assessment_tab")]

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        yield [gr.update(value=f"<h3 style='color:#ef4444;'>⚠️ Error: {str(e)}</h3>", visible=True),
               None, []] + [gr.update(visible=False)] * MAX_QUESTIONS + [gr.update(selected="upload_tab")]


# ═══════════════════════════════════════════════════════════════════
# Grade & Match — now returns 3 Phase 1 outputs
# ═══════════════════════════════════════════════════════════════════
def grade_and_match(user_id, quiz_state, email_for_report, *user_answers):
    if not quiz_state:
        return "", "", pd.DataFrame(), "", "", None, {}, gr.update(selected="results_tab")

    review_html     = build_review_html(quiz_state, user_answers)
    verified_scores = grade_assessment(user_answers, quiz_state)
    user_vector     = [verified_scores.get(skill, 0) for skill in skills_list]
    user_vector_df  = pd.DataFrame([user_vector], columns=skills_list)

    # ── ML Model 1: KNN Job Matching ────────────────────────────────
    knn_matches   = _knn_model.predict(verified_scores, top_n=3)
    best_job      = knn_matches[0]['job'] if knn_matches else "Unknown"
    missing_skills= knn_matches[0]['missing'] if knn_matches else []

    # top3_jobs for PDF and salary
    top3_jobs = [(m['job'], m['confidence']) for m in knn_matches]

    total   = len(quiz_state)
    correct = sum(1 for i, item in enumerate(quiz_state)
                  if i < len(user_answers) and user_answers[i]
                  and str(user_answers[i]).strip().upper()[0] == str(item.get("correct_answer","")).strip().upper())
    quiz_pct = round((correct / total) * 100) if total else 0

    if user_id:
        try:
            save_verified_skills(user_id, verified_scores)
            save_recommendation(user_id, best_job, missing_skills, quiz_pct)
        except Exception as e:
            print(f"Database save error: {e}")

    score_data = [[k.title(), f"{v} / 5"] for k, v in verified_scores.items()]
    score_df   = pd.DataFrame(score_data, columns=["Technical Skill", "Verified Score"])

    # ── ML Model 1 HTML: KNN Top-3 matches ──────────────────────────
    top3_html = build_knn_matches_html(verified_scores)

    # ── ML Model 2 HTML: Regression Skill Forecast ──────────────────
    user_skills_with_scores = [s for s, v in verified_scores.items() if v > 0]
    trend_html = build_forecast_html(user_skills_with_scores)

    # ── Learning path (uses KNN best match gap skills) ───────────────
    learning_html = build_learning_path_html(missing_skills, best_job)

    # PDF report
    pdf_path = None
    try:
        import datetime, tempfile
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(tempfile.gettempdir(), f"career_report_{ts}.pdf")
        generate_pdf_report(
            email          = email_for_report or "user@example.com",
            best_job       = str(best_job),
            top3_jobs      = top3_jobs,
            score_df       = score_df,
            missing_skills = missing_skills,
            verified_scores= verified_scores,
            quiz_pct       = quiz_pct,
            output_path    = pdf_path
        )
        print(f"✅ PDF ready: {pdf_path}")
    except Exception as e:
        print(f"PDF generation error: {e}")
        import traceback; traceback.print_exc()
        pdf_path = None

    return (review_html, top3_html, score_df, trend_html, learning_html,
            pdf_path, verified_scores, gr.update(selected="results_tab"))


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — FEATURE 1: User Dashboard
# ═══════════════════════════════════════════════════════════════════
def load_dashboard(email):
    if not email or "@" not in email:
        return build_empty_dashboard("Please enter a valid email address.")

    rec_history, latest_skills, progress_rows = get_user_dashboard(email)

    if rec_history is None:
        return build_empty_dashboard(f"No account found for <b>{email}</b>. Complete an assessment first.")

    # ── Assessment history ──
    hist_html = """
    <div style='margin-bottom:24px;'>
        <h3 style='color:#10b981;font-size:1em;font-weight:700;margin-bottom:12px;'>
            📋 Assessment History
        </h3>"""

    if rec_history:
        for role, pct, missing, date in rec_history:
            pct = pct or 0
            color = "#10b981" if pct >= 70 else "#f59e0b" if pct >= 40 else "#ef4444"
            date_str = date.strftime("%b %d, %Y  %H:%M") if date else "—"
            hist_html += f"""
            <div style='padding:12px 16px;border-radius:8px;border:1px solid #1e293b;
                        background:#0f172a;margin-bottom:8px;
                        display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <p style='color:#e2e8f0;font-weight:700;font-size:0.9em;margin:0;'>💼 {role}</p>
                    <p style='color:#475569;font-size:0.78em;margin:3px 0 0;'>🗓️ {date_str}</p>
                </div>
                <div style='text-align:right;'>
                    <span style='color:{color};font-weight:800;font-size:1.1em;'>{pct}%</span>
                    <div style='width:80px;background:#1e293b;border-radius:4px;height:5px;margin-top:4px;'>
                        <div style='width:{pct}%;background:{color};height:5px;border-radius:4px;'></div>
                    </div>
                </div>
            </div>"""
    else:
        hist_html += "<p style='color:#475569;font-size:0.85em;'>No assessments yet.</p>"

    hist_html += "</div>"

    # ── Latest skill scores ──
    skills_html = """
    <div style='margin-bottom:24px;'>
        <h3 style='color:#3b82f6;font-size:1em;font-weight:700;margin-bottom:12px;'>
            🧠 Current Skill Levels
        </h3>
        <div style='display:flex;flex-wrap:wrap;gap:8px;'>"""

    if latest_skills:
        for skill, score, date in latest_skills:
            colors_map = ["#ef4444","#f87171","#f59e0b","#10b981","#059669","#047857"]
            sc = colors_map[min(score, 5)]
            skills_html += f"""
            <div style='padding:8px 14px;border-radius:8px;border:1px solid #1e293b;
                        background:#0f172a;text-align:center;min-width:110px;'>
                <p style='color:#94a3b8;font-size:0.75em;margin:0 0 3px;'>{skill.title()}</p>
                <p style='color:{sc};font-weight:800;font-size:1.1em;margin:0;'>{score}/5</p>
            </div>"""
    else:
        skills_html += "<p style='color:#475569;font-size:0.85em;'>No skills recorded yet.</p>"

    skills_html += "</div></div>"

    return f"""
    <div style='padding:16px;border-radius:12px;border:1px solid #1e293b;background:#0f172a;'>
        <div style='margin-bottom:20px;padding:14px;border-radius:8px;
                    background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #10b981;'>
            <p style='color:#10b981;font-weight:700;margin:0;font-size:1em;'>👤 {email}</p>
            <p style='color:#64748b;font-size:0.8em;margin:4px 0 0;'>
                {len(rec_history)} assessment(s) completed · {len(latest_skills or [])} skills tracked
            </p>
        </div>
        {hist_html}
        {skills_html}
    </div>"""


def build_empty_dashboard(msg):
    return f"""
    <div style='padding:40px;border-radius:12px;border:1px solid #1e293b;
                background:#0f172a;text-align:center;'>
        <div style='font-size:2.5em;margin-bottom:12px;'>📊</div>
        <p style='color:#94a3b8;font-size:0.9em;'>{msg}</p>
    </div>"""


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — FEATURE 3: Skill Progress Chart
# ═══════════════════════════════════════════════════════════════════
def build_progress_chart(email):
    if not email or "@" not in email:
        return None

    _, latest_skills, _ = get_user_dashboard(email)
    if not latest_skills:
        return None

    skills = [row[0].title() for row in latest_skills[:12]]
    scores = [row[1] for row in latest_skills[:12]]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    colors = ["#10b981" if s >= 4 else "#f59e0b" if s >= 2 else "#ef4444" for s in scores]
    bars   = ax.barh(skills, scores, color=colors, height=0.55, zorder=3)

    # Score labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{score}/5", va="center", ha="left",
                color="#e2e8f0", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 6)
    ax.set_xlabel("Verified Score (0–5)", color="#64748b", fontsize=9)
    ax.set_title(f"Skill Scores — {email}", color="#e2e8f0", fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines[:].set_color("#1e293b")
    ax.axvline(x=3, color="#334155", linestyle="--", linewidth=1, zorder=2, label="Proficiency threshold")
    ax.grid(axis="x", color="#1e293b", linewidth=0.5, zorder=1)

    legend = [
        mpatches.Patch(color="#10b981", label="Advanced (4-5)"),
        mpatches.Patch(color="#f59e0b", label="Intermediate (2-3)"),
        mpatches.Patch(color="#ef4444", label="Beginner (0-1)"),
    ]
    ax.legend(handles=legend, loc="lower right", facecolor="#1e293b",
              edgecolor="#334155", labelcolor="#94a3b8", fontsize=8)

    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    return tmp.name


# ═══════════════════════════════════════════════════════════════════
# PHASE 3 — FEATURE 1: Resume Improvement Suggestions
# ═══════════════════════════════════════════════════════════════════
def generate_resume_suggestions(pdf_path, target_job, api_key):
    if not pdf_path:
        return "<p style='color:#ef4444;'>⚠️ Please upload your resume first.</p>"

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return "<p style='color:#ef4444;'>⚠️ GROQ_API_KEY not found in .env</p>"

    try:
        from groq import Groq
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            return "<p style='color:#ef4444;'>⚠️ Could not extract text from PDF.</p>"

        client = Groq(api_key=groq_key)
        prompt = f"""You are an expert resume coach and technical recruiter.

Analyze this resume text and provide specific, actionable improvement suggestions.
{"Target job role: " + target_job if target_job else ""}

Resume text (first 3000 chars):
{text[:3000]}

Provide your response in this exact JSON format:
{{
  "overall_score": <number 1-10>,
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "improvements": [
    {{"section": "section name", "issue": "what's wrong", "fix": "how to fix it"}},
    {{"section": "section name", "issue": "what's wrong", "fix": "how to fix it"}},
    {{"section": "section name", "issue": "what's wrong", "fix": "how to fix it"}}
  ],
  "missing_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "quick_wins": ["quick win 1", "quick win 2", "quick win 3"]
}}

Return only raw JSON, no markdown."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content
        import json, re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return "<p style='color:#ef4444;'>⚠️ AI response parsing failed. Try again.</p>"

        data = json.loads(match.group(0))
        score    = data.get("overall_score", 5)
        strengths = data.get("strengths", [])
        improvements = data.get("improvements", [])
        keywords = data.get("missing_keywords", [])
        quick_wins = data.get("quick_wins", [])

        score_color = "#10b981" if score >= 7 else "#f59e0b" if score >= 5 else "#ef4444"

        # Build HTML
        html = f"""
        <div style='padding:16px;border-radius:12px;border:1px solid #1e293b;background:#0f172a;'>

            <div style='display:flex;align-items:center;gap:16px;padding:16px;border-radius:10px;
                        background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid {score_color};
                        margin-bottom:20px;'>
                <div style='text-align:center;min-width:80px;'>
                    <p style='color:{score_color};font-size:2.5em;font-weight:800;margin:0;'>{score}</p>
                    <p style='color:#64748b;font-size:0.75em;margin:0;'>out of 10</p>
                </div>
                <div>
                    <p style='color:#e2e8f0;font-weight:700;font-size:1em;margin:0 0 4px;'>Resume Score</p>
                    <p style='color:#64748b;font-size:0.82em;margin:0;'>
                        {"Excellent — ready to apply! 🚀" if score >= 8 else
                         "Good — a few tweaks needed 🔧" if score >= 6 else
                         "Needs improvement — follow suggestions below 📝"}
                    </p>
                </div>
            </div>

            <div style='margin-bottom:18px;'>
                <p style='color:#10b981;font-weight:700;font-size:0.88em;margin-bottom:8px;'>✅ Strengths</p>
                {"".join([f"<div style='padding:8px 12px;border-radius:7px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);color:#cbd5e1;font-size:0.85em;margin-bottom:6px;'>✓ {s}</div>" for s in strengths])}
            </div>

            <div style='margin-bottom:18px;'>
                <p style='color:#ef4444;font-weight:700;font-size:0.88em;margin-bottom:8px;'>🔧 Improvement Areas</p>
                {"".join([f'''<div style='padding:12px;border-radius:8px;border:1px solid #1e293b;
                    background:#0a1628;margin-bottom:8px;border-left:3px solid #ef4444;'>
                    <p style='color:#f87171;font-weight:700;font-size:0.82em;margin:0 0 4px;'>
                        📌 {imp.get("section","").upper()}</p>
                    <p style='color:#94a3b8;font-size:0.82em;margin:0 0 4px;'>❌ {imp.get("issue","")}</p>
                    <p style='color:#10b981;font-size:0.82em;margin:0;'>✅ {imp.get("fix","")}</p>
                </div>''' for imp in improvements])}
            </div>

            <div style='margin-bottom:18px;'>
                <p style='color:#f59e0b;font-weight:700;font-size:0.88em;margin-bottom:8px;'>🔑 Missing Keywords to Add</p>
                <div>{"".join([f"<span style='display:inline-block;margin:3px;padding:4px 12px;border-radius:14px;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3);color:#fbbf24;font-size:0.82em;font-weight:600;'>+{k}</span>" for k in keywords])}</div>
            </div>

            <div>
                <p style='color:#8b5cf6;font-weight:700;font-size:0.88em;margin-bottom:8px;'>⚡ Quick Wins (Do These First)</p>
                {"".join([f"<div style='padding:8px 12px;border-radius:7px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);color:#cbd5e1;font-size:0.85em;margin-bottom:6px;'>→ {qw}</div>" for qw in quick_wins])}
            </div>
        </div>"""
        return html

    except Exception as e:
        return f"<p style='color:#ef4444;'>⚠️ Error: {str(e)}</p>"


# ═══════════════════════════════════════════════════════════════════
# PHASE 3 — FEATURE 2: Job Description Matcher
# ═══════════════════════════════════════════════════════════════════
def match_job_description(jd_text, verified_scores_state):
    if not jd_text or not jd_text.strip():
        return "<p style='color:#ef4444;'>⚠️ Please paste a job description.</p>"
    if not verified_scores_state:
        return "<p style='color:#ef4444;'>⚠️ Please complete an assessment first so we know your skills.</p>"

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return "<p style='color:#ef4444;'>⚠️ GROQ_API_KEY not found in .env</p>"

    try:
        from groq import Groq
        import json, re

        user_skills_str = ", ".join([f"{k} ({v}/5)" for k, v in verified_scores_state.items()])

        client = Groq(api_key=groq_key)
        prompt = f"""You are a technical recruiter. Analyze the fit between a candidate's skills and a job description.

Candidate's verified skills: {user_skills_str}

Job Description:
{jd_text[:3000]}

Respond in this exact JSON format:
{{
  "match_score": <number 0-100>,
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "nice_to_have": ["skill1", "skill2"],
  "verdict": "one sentence overall verdict",
  "apply_recommendation": "YES" or "NO" or "MAYBE"
}}

Return only raw JSON."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        raw = response.choices[0].message.content
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return "<p style='color:#ef4444;'>⚠️ Could not parse AI response.</p>"

        data = json.loads(match.group(0))
        score    = data.get("match_score", 0)
        matched  = data.get("matched_skills", [])
        missing  = data.get("missing_skills", [])
        nice     = data.get("nice_to_have", [])
        verdict  = data.get("verdict", "")
        apply    = data.get("apply_recommendation", "MAYBE")

        score_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"
        apply_styles = {
            "YES":   ("✅ APPLY NOW",   "#10b981", "rgba(16,185,129,0.1)"),
            "MAYBE": ("⚠️ MAYBE APPLY", "#f59e0b", "rgba(245,158,11,0.1)"),
            "NO":    ("❌ NOT YET",     "#ef4444", "rgba(239,68,68,0.1)"),
        }
        apply_label, apply_color, apply_bg = apply_styles.get(apply, apply_styles["MAYBE"])

        html = f"""
        <div style='padding:16px;border-radius:12px;border:1px solid #1e293b;background:#0f172a;'>

            <div style='display:flex;gap:14px;margin-bottom:20px;'>
                <div style='flex:1;padding:16px;border-radius:10px;border:1px solid {score_color};
                            background:rgba(0,0,0,0.2);text-align:center;'>
                    <p style='color:{score_color};font-size:2.5em;font-weight:800;margin:0;'>{score}%</p>
                    <p style='color:#64748b;font-size:0.78em;margin:0;'>JD Match Score</p>
                    <div style='width:100%;background:#1e293b;border-radius:4px;height:6px;margin-top:8px;'>
                        <div style='width:{score}%;background:{score_color};height:6px;border-radius:4px;'></div>
                    </div>
                </div>
                <div style='flex:1;padding:16px;border-radius:10px;background:{apply_bg};
                            border:1px solid {apply_color};text-align:center;display:flex;
                            flex-direction:column;justify-content:center;'>
                    <p style='color:{apply_color};font-size:1.1em;font-weight:800;margin:0 0 6px;'>{apply_label}</p>
                    <p style='color:#94a3b8;font-size:0.78em;margin:0;'>{verdict}</p>
                </div>
            </div>

            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;'>
                <div style='padding:14px;border-radius:8px;border:1px solid rgba(16,185,129,0.3);background:rgba(16,185,129,0.05);'>
                    <p style='color:#10b981;font-weight:700;font-size:0.85em;margin-bottom:8px;'>✅ Skills You Have ({len(matched)})</p>
                    {"".join([f"<span style='display:inline-block;margin:2px;padding:3px 10px;border-radius:12px;background:rgba(16,185,129,0.15);color:#10b981;font-size:0.78em;'>{s}</span>" for s in matched])}
                </div>
                <div style='padding:14px;border-radius:8px;border:1px solid rgba(239,68,68,0.3);background:rgba(239,68,68,0.05);'>
                    <p style='color:#ef4444;font-weight:700;font-size:0.85em;margin-bottom:8px;'>❌ Skills to Learn ({len(missing)})</p>
                    {"".join([f"<span style='display:inline-block;margin:2px;padding:3px 10px;border-radius:12px;background:rgba(239,68,68,0.12);color:#f87171;font-size:0.78em;'>+{s}</span>" for s in missing])}
                </div>
            </div>

            {"" if not nice else f'''<div style="padding:12px;border-radius:8px;border:1px solid #334155;background:#0a1628;">
                <p style="color:#64748b;font-weight:700;font-size:0.82em;margin-bottom:6px;">💡 Nice to Have</p>
                {"".join([f'<span style="display:inline-block;margin:2px;padding:3px 10px;border-radius:12px;background:#1e293b;color:#64748b;font-size:0.78em;">{s}</span>' for s in nice])}
            </div>'''}
        </div>"""
        return html

    except Exception as e:
        return f"<p style='color:#ef4444;'>⚠️ Error: {str(e)}</p>"


# ═══════════════════════════════════════════════════════════════════
# PHASE 3 — FEATURE 3: Salary Insights
# ═══════════════════════════════════════════════════════════════════
SALARY_DATA = {
    "data scientist":              {"min": 90,  "mid": 120, "max": 160, "currency": "USD", "demand": "Very High"},
    "machine learning engineer":   {"min": 100, "mid": 135, "max": 175, "currency": "USD", "demand": "Very High"},
    "software engineer":           {"min": 85,  "mid": 115, "max": 155, "currency": "USD", "demand": "Very High"},
    "devops engineer":             {"min": 90,  "mid": 120, "max": 155, "currency": "USD", "demand": "High"},
    "data engineer":               {"min": 90,  "mid": 120, "max": 155, "currency": "USD", "demand": "High"},
    "cloud architect":             {"min": 110, "mid": 145, "max": 185, "currency": "USD", "demand": "High"},
    "cybersecurity engineer":      {"min": 85,  "mid": 115, "max": 155, "currency": "USD", "demand": "Very High"},
    "full stack developer":        {"min": 75,  "mid": 100, "max": 135, "currency": "USD", "demand": "High"},
    "frontend developer":          {"min": 65,  "mid": 90,  "max": 125, "currency": "USD", "demand": "High"},
    "backend developer":           {"min": 70,  "mid": 100, "max": 135, "currency": "USD", "demand": "High"},
    "data analyst":                {"min": 60,  "mid": 80,  "max": 110, "currency": "USD", "demand": "High"},
    "business intelligence analyst":{"min": 60, "mid": 80,  "max": 110, "currency": "USD", "demand": "Medium"},
    "network engineer":            {"min": 65,  "mid": 85,  "max": 115, "currency": "USD", "demand": "Medium"},
    "database administrator":      {"min": 70,  "mid": 95,  "max": 130, "currency": "USD", "demand": "Medium"},
    "site reliability engineer":   {"min": 95,  "mid": 130, "max": 170, "currency": "USD", "demand": "High"},
    "ai engineer":                 {"min": 105, "mid": 140, "max": 185, "currency": "USD", "demand": "Very High"},
    "nlp engineer":                {"min": 100, "mid": 135, "max": 170, "currency": "USD", "demand": "High"},
    "platform engineer":           {"min": 90,  "mid": 120, "max": 158, "currency": "USD", "demand": "High"},
    "solutions architect":         {"min": 105, "mid": 140, "max": 180, "currency": "USD", "demand": "High"},
    "it manager":                  {"min": 85,  "mid": 115, "max": 150, "currency": "USD", "demand": "Medium"},
}

def build_salary_html(best_job, top3_jobs_state):
    """Shows salary insights for the top matched job roles."""
    if not top3_jobs_state:
        return "<p style='color:#64748b;font-size:0.85em;'>Complete an assessment to see salary insights.</p>"

    html = """
    <div>
        <h3 style='color:#f59e0b;font-size:1em;font-weight:700;margin-bottom:4px;'>
            💰 Salary Insights for Your Top Matches
        </h3>
        <p style='color:#475569;font-size:0.78em;margin-bottom:16px;'>
            Based on global tech market data (USD/year). Figures represent base salary ranges.
        </p>"""

    demand_color = {"Very High": "#10b981", "High": "#3b82f6", "Medium": "#f59e0b", "Low": "#ef4444"}

    for job_name, match_pct in top3_jobs_state:
        key = job_name.lower().strip()
        # fuzzy match
        data = SALARY_DATA.get(key)
        if not data:
            for k, v in SALARY_DATA.items():
                if k in key or any(word in key for word in k.split()):
                    data = v
                    break
        if not data:
            data = {"min": 65, "mid": 90, "max": 120, "demand": "Medium"}

        mn, md, mx = data["min"], data["mid"], data["max"]
        demand     = data.get("demand", "Medium")
        dc         = demand_color.get(demand, "#64748b")

        # bar widths relative to $200k max
        bar_min = round((mn / 200) * 100)
        bar_mid = round((md / 200) * 100)
        bar_max = round((mx / 200) * 100)

        html += f"""
        <div style='margin-bottom:16px;padding:16px;border-radius:10px;border:1px solid #1e293b;background:#0a1628;'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
                <div>
                    <p style='color:#e2e8f0;font-weight:700;font-size:0.92em;margin:0;'>💼 {job_name}</p>
                    <p style='color:#475569;font-size:0.75em;margin:3px 0 0;'>JD Match: {match_pct}%</p>
                </div>
                <span style='background:{dc};color:white;padding:3px 10px;border-radius:12px;
                             font-size:0.72em;font-weight:700;'>{demand} Demand</span>
            </div>

            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;'>
                <div style='padding:10px;border-radius:7px;background:#0f172a;text-align:center;border:1px solid #1e293b;'>
                    <p style='color:#64748b;font-size:0.72em;margin:0 0 3px;'>Entry Level</p>
                    <p style='color:#94a3b8;font-weight:700;font-size:0.95em;margin:0;'>${mn}K</p>
                </div>
                <div style='padding:10px;border-radius:7px;background:#0f172a;text-align:center;border:1px solid #10b981;'>
                    <p style='color:#10b981;font-size:0.72em;margin:0 0 3px;font-weight:600;'>Mid Level ⭐</p>
                    <p style='color:#10b981;font-weight:800;font-size:1.1em;margin:0;'>${md}K</p>
                </div>
                <div style='padding:10px;border-radius:7px;background:#0f172a;text-align:center;border:1px solid #1e293b;'>
                    <p style='color:#64748b;font-size:0.72em;margin:0 0 3px;'>Senior Level</p>
                    <p style='color:#e2e8f0;font-weight:700;font-size:0.95em;margin:0;'>${mx}K</p>
                </div>
            </div>

            <div style='position:relative;height:8px;background:#1e293b;border-radius:6px;'>
                <div style='position:absolute;left:{bar_min}%;width:{bar_max - bar_min}%;
                            height:8px;background:linear-gradient(to right,#3b82f6,#10b981);
                            border-radius:6px;opacity:0.6;'></div>
                <div style='position:absolute;left:{bar_mid - 1}%;width:3%;
                            height:8px;background:#f59e0b;border-radius:6px;'></div>
            </div>
            <div style='display:flex;justify-content:space-between;margin-top:4px;'>
                <span style='color:#475569;font-size:0.7em;'>${mn}K</span>
                <span style='color:#f59e0b;font-size:0.7em;font-weight:600;'>${md}K avg</span>
                <span style='color:#475569;font-size:0.7em;'>${mx}K</span>
            </div>
        </div>"""

    html += """
        <p style='color:#334155;font-size:0.72em;margin-top:8px;text-align:center;'>
            💡 Salaries vary by location, company size, and experience. Source: Market averages 2024-2025.
        </p>
    </div>"""
    return html


# ═══════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════
professional_theme = gr.themes.Soft(
    primary_hue="emerald", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
)

with gr.Blocks() as app:
    gr.HTML("""<div class="app-header">
        <h1>🚀 Skill Gap Navigator</h1>
        <p>Upload your resume · Verify skills · Get your personalized career roadmap</p>
    </div>""")

    with gr.Accordion("⚙️ System Settings (Required)", open=False):
        api_key_input = gr.Textbox(label="🔑 API Key", type="password",
                                   value=os.getenv("GEMINI_API_KEY", ""),
                                   placeholder="Enter your API key...")

    with gr.Tabs(elem_id="main_tabs") as tabs:

        # ── Tab 1: Profile Upload ──────────────────────────────────────────
        with gr.TabItem("📄  1. Profile Upload", id="upload_tab"):
            gr.HTML("""<div style='padding:14px 0 6px;'>
                <h2 style='margin:0;font-size:1.3em;'>Build Your Profile</h2>
                <p style='color:#94a3b8;margin:4px 0 0;font-size:0.88em;'>
                    Add your resume and optionally attach certificates with their skills.</p></div>""")

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    gr.HTML("<p style='color:#10b981;font-weight:700;margin:8px 0 6px;font-size:0.88em;'>👤 BASIC INFO</p>")
                    email_input  = gr.Textbox(label="Email Address *", placeholder="you@example.com")
                    resume_input = gr.File(label="📎 PDF Resume *", type="filepath")

                    gr.HTML("""<div style='border-top:1px solid #1e293b;margin-top:18px;padding-top:16px;'>
                        <p style='color:#60a5fa;font-weight:700;margin:0 0 4px;font-size:0.88em;'>📜 CERTIFICATES</p>
                        <p style='color:#475569;font-size:0.78em;margin:0 0 12px;'>
                            Each card = one certificate. Add name, skills, and optionally upload the PDF.</p></div>""")

                    cert_count       = gr.State(1)
                    cert_card_rows   = []
                    cert_name_boxes  = []
                    cert_skill_boxes = []
                    cert_file_boxes  = []

                    for i in range(MAX_CERTS):
                        with gr.Group(visible=(i == 0), elem_classes="cert-card") as card_row:
                            gr.HTML(f"<p style='color:#60a5fa;font-weight:700;font-size:0.85em;margin:0 0 10px;'>🎓 Certificate {i+1}</p>")
                            name_box  = gr.Textbox(label="Certificate Name", placeholder="e.g. AWS Certified Developer...", lines=1)
                            skill_box = gr.Textbox(label="Skills it covers (comma separated)", placeholder="e.g. Python, Machine Learning, SQL...", lines=2)
                            file_box  = gr.File(label="Upload Certificate PDF (optional)", type="filepath", file_types=[".pdf"])
                        cert_card_rows.append(card_row)
                        cert_name_boxes.append(name_box)
                        cert_skill_boxes.append(skill_box)
                        cert_file_boxes.append(file_box)

                    with gr.Row():
                        add_btn    = gr.Button("＋ Add Certificate", elem_classes="btn-add",    size="sm")
                        remove_btn = gr.Button("－ Remove Last",     elem_classes="btn-remove",  size="sm")

                    gr.HTML("<div style='height:14px;'></div>")
                    extract_btn = gr.Button("📄 Parse Resume & Start Assessment",
                                            variant="primary", size="lg", elem_classes="btn-primary")

                with gr.Column(scale=2):
                    skills_output = gr.HTML(
                        value="""<div style='padding:28px 20px;border-radius:12px;border:1px solid #1e293b;
                                    background:#0f172a;text-align:center;min-height:350px;'>
                            <div style='font-size:2.5em;margin-bottom:14px;'>📋</div>
                            <p style='color:#cbd5e1;font-weight:600;font-size:1em;margin-bottom:8px;'>Profile Summary</p>
                            <p style='color:#475569;font-size:0.82em;line-height:1.7;'>
                                Fill in your details on the left,<br>then click <b style='color:#10b981;'>Parse Resume</b>.</p>
                        </div>""", visible=True)
            user_id_state = gr.State(None)

        # ── Tab 2: Skill Verification ──────────────────────────────────────
        with gr.TabItem("📝  2. Skill Verification", id="assessment_tab"):
            gr.HTML("""<div style='padding:14px 0 8px;'>
                <h2 style='margin:0;font-size:1.3em;'>📝 Technical Assessment</h2>
                <p style='color:#94a3b8;margin:4px 0 0;font-size:0.88em;'>
                    Answer honestly — your score determines the skill verification rating.</p></div>""")
            quiz_state       = gr.State([])
            radio_components = []
            with gr.Column():
                for i in range(MAX_QUESTIONS):
                    radio = gr.Radio(choices=[], label="", visible=False, elem_classes="question-box")
                    radio_components.append(radio)
            gr.HTML("<hr style='border-color:#1e293b;margin:20px 0;'>")
            grade_btn = gr.Button("🎯 Submit Assessment & See Results",
                                  variant="primary", size="lg", elem_classes="btn-primary")

        # ── Tab 3: Results (Phase 1 + Phase 2 PDF) ────────────────────────
        with gr.TabItem("🏆  3. Analysis Report", id="results_tab"):
            gr.HTML("""<div style='padding:14px 0 8px;'>
                <h2 style='margin:0;font-size:1.3em;'>🏆 Your Personalized Career Analysis</h2>
                <p style='color:#94a3b8;margin:4px 0 0;font-size:0.88em;'>
                    Based on your resume, certificates, and assessment performance.</p></div>""")

            # Phase 2 — PDF download
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML("<p style='color:#64748b;font-size:0.82em;margin:0;'>📄 Your full career report is auto-generated after you submit the assessment.</p>")
                with gr.Column(scale=1):
                    pdf_download = gr.File(
                        label="⬇️ Download PDF Report",
                        visible=True,
                        interactive=False,
                        file_count="single"
                    )

            gr.HTML("<hr style='border-color:#1e293b;margin:16px 0;'>")

            # Answer review
            review_output = gr.HTML(value="")
            gr.HTML("<hr style='border-color:#1e293b;margin:20px 0;'>")

            # Top 3 job matches
            top3_output = gr.HTML(value="")
            gr.HTML("<hr style='border-color:#1e293b;margin:20px 0;'>")

            # Skill scores
            gr.Markdown("### 🧠 Your Verified Skill Scores")
            score_table = gr.Dataframe(headers=["Technical Skill", "Verified Score"],
                                       interactive=False, wrap=True)
            gr.HTML("<hr style='border-color:#1e293b;margin:20px 0;'>")

            # Trend analysis
            trend_output = gr.HTML(value="")
            gr.HTML("<hr style='border-color:#1e293b;margin:20px 0;'>")

            # Learning path
            learning_output = gr.HTML(value="")

        # ── Tab 4: Dashboard (Phase 2) ─────────────────────────────────────
        with gr.TabItem("📊  4. My Dashboard", id="dashboard_tab"):
            gr.HTML("""<div style='padding:14px 0 8px;'>
                <h2 style='margin:0;font-size:1.3em;'>📊 Your Learning Dashboard</h2>
                <p style='color:#94a3b8;margin:4px 0 0;font-size:0.88em;'>
                    Track your skill progress and assessment history over time.</p></div>""")

            with gr.Row():
                dash_email = gr.Textbox(label="Enter your email to load dashboard",
                                        placeholder="you@example.com", scale=4)
                dash_btn   = gr.Button("Load Dashboard", variant="primary", scale=1,
                                       elem_classes="btn-primary")

            dashboard_output = gr.HTML(value="""
            <div style='padding:40px;border-radius:12px;border:1px solid #1e293b;
                        background:#0f172a;text-align:center;'>
                <div style='font-size:2.5em;margin-bottom:12px;'>📊</div>
                <p style='color:#94a3b8;'>Enter your email above and click Load Dashboard.</p>
            </div>""")

            gr.HTML("<hr style='border-color:#1e293b;margin:20px 0;'>")

            # Phase 2 Feature 3 — Progress Chart
            gr.HTML("<h3 style='color:#8b5cf6;font-size:1em;font-weight:700;margin-bottom:12px;'>📈 Skill Progress Chart</h3>")
            progress_chart = gr.Image(label="", show_label=False, visible=True)
            chart_btn = gr.Button("🔄 Generate Progress Chart", size="sm", elem_classes="btn-add")

        # ── Tab 5: AI Tools (Phase 3) ──────────────────────────────────────
        with gr.TabItem("🤖  5. AI Tools", id="ai_tools_tab"):
            gr.HTML("""<div style='padding:14px 0 8px;'>
                <h2 style='margin:0;font-size:1.3em;'>🤖 Advanced AI Career Tools</h2>
                <p style='color:#94a3b8;margin:4px 0 0;font-size:0.88em;'>
                    AI-powered tools to supercharge your job search.</p></div>""")

            with gr.Tabs():

                # ── Sub-tab A: Resume Suggestions ──────────────────────────
                with gr.TabItem("📝 Resume Improvement"):
                    gr.HTML("""<div style='padding:10px 0 6px;'>
                        <p style='color:#94a3b8;font-size:0.85em;margin:0;'>
                            Upload your resume and get AI-powered suggestions to improve it.
                            Optionally enter a target job role for tailored advice.
                        </p></div>""")
                    with gr.Row():
                        with gr.Column(scale=1):
                            resume_ai_input  = gr.File(label="📎 Upload Resume PDF", type="filepath")
                            target_job_input = gr.Textbox(
                                label="Target Job Role (optional)",
                                placeholder="e.g. Data Scientist, DevOps Engineer..."
                            )
                            resume_btn = gr.Button("🤖 Analyse My Resume",
                                                   variant="primary", elem_classes="btn-primary")
                        with gr.Column(scale=2):
                            resume_suggestions_output = gr.HTML(value="""
                            <div style='padding:30px;border-radius:12px;border:1px solid #1e293b;
                                        background:#0f172a;text-align:center;'>
                                <div style='font-size:2em;margin-bottom:10px;'>📝</div>
                                <p style='color:#94a3b8;font-size:0.88em;'>
                                    Upload your resume and click Analyse to get suggestions.</p>
                            </div>""")

                # ── Sub-tab B: JD Matcher ───────────────────────────────────
                with gr.TabItem("📋 Job Description Matcher"):
                    gr.HTML("""<div style='padding:10px 0 6px;'>
                        <p style='color:#94a3b8;font-size:0.85em;margin:0;'>
                            Paste any job description and instantly see how well your skills match.
                            Complete an assessment first so we know your skill levels.
                        </p></div>""")
                    with gr.Row():
                        with gr.Column(scale=1):
                            jd_input = gr.Textbox(
                                label="Paste Job Description Here",
                                placeholder="Copy and paste the full job description...",
                                lines=12, max_lines=20
                            )
                            jd_btn = gr.Button("🔍 Match My Skills to This JD",
                                               variant="primary", elem_classes="btn-primary")
                        with gr.Column(scale=1):
                            jd_output = gr.HTML(value="""
                            <div style='padding:30px;border-radius:12px;border:1px solid #1e293b;
                                        background:#0f172a;text-align:center;'>
                                <div style='font-size:2em;margin-bottom:10px;'>📋</div>
                                <p style='color:#94a3b8;font-size:0.88em;'>
                                    Paste a JD and click Match to see your fit score.</p>
                            </div>""")

                # ── Sub-tab C: Salary Insights ──────────────────────────────
                with gr.TabItem("💰 Salary Insights"):
                    gr.HTML("""<div style='padding:10px 0 6px;'>
                        <p style='color:#94a3b8;font-size:0.85em;margin:0;'>
                            See salary ranges for your top matched job roles.
                            Complete an assessment first to populate your matches.
                        </p></div>""")
                    salary_output = gr.HTML(value="""
                    <div style='padding:30px;border-radius:12px;border:1px solid #1e293b;
                                background:#0f172a;text-align:center;'>
                        <div style='font-size:2em;margin-bottom:10px;'>💰</div>
                        <p style='color:#94a3b8;font-size:0.88em;'>
                            Complete your assessment first, then come back here to see salary insights.</p>
                    </div>""")
                    salary_btn = gr.Button("💰 Load Salary Insights",
                                           variant="primary", elem_classes="btn-primary")

    # ── States ───────────────────────────────────────────────────────────────
    verified_scores_state = gr.State({})
    top3_jobs_state       = gr.State([])

    # ── Handlers ────────────────────────────────────────────────────────────
    add_btn.click(add_cert, inputs=[cert_count], outputs=[cert_count] + cert_card_rows)
    remove_btn.click(remove_cert, inputs=[cert_count], outputs=[cert_count] + cert_card_rows)

    extract_btn.click(
        process_and_quiz,
        inputs=[email_input, resume_input, api_key_input]
              + cert_name_boxes + cert_skill_boxes + cert_file_boxes,
        outputs=[skills_output, user_id_state, quiz_state] + radio_components + [tabs],
        concurrency_limit=4, show_progress="full"
    )

    email_input.change(fn=lambda e: e, inputs=[email_input], outputs=[dash_email])

    grade_btn.click(
        grade_and_match,
        inputs=[user_id_state, quiz_state, email_input] + radio_components,
        outputs=[review_output, top3_output, score_table, trend_output,
                 learning_output, pdf_download, verified_scores_state, tabs],
        concurrency_limit=4
    )

    # Phase 3 handlers
    resume_btn.click(
        generate_resume_suggestions,
        inputs=[resume_ai_input, target_job_input, api_key_input],
        outputs=[resume_suggestions_output]
    )

    jd_btn.click(
        match_job_description,
        inputs=[jd_input, verified_scores_state],
        outputs=[jd_output]
    )

    # Salary — compute top3 on the fly from state
    def load_salary(verified_scores):
        if not verified_scores:
            return build_salary_html("", [])
        user_vector    = [verified_scores.get(skill, 0) for skill in skills_list]
        user_vector_df = pd.DataFrame([user_vector], columns=skills_list)
        sim_scores     = cosine_similarity(user_vector_df, skill_matrix)[0]
        top3_idx       = np.argsort(sim_scores)[::-1][:3]
        top3           = [(job_titles.iloc[int(i)], min(round(float(sim_scores[i])*100), 99)) for i in top3_idx]
        return build_salary_html(str(top3[0][0]), top3)

    salary_btn.click(load_salary, inputs=[verified_scores_state], outputs=[salary_output])

    dash_btn.click(load_dashboard, inputs=[dash_email], outputs=[dashboard_output])
    chart_btn.click(build_progress_chart, inputs=[dash_email], outputs=[progress_chart])

app.queue(max_size=5)

if __name__ == "__main__":
    # Get the port from Render (defaults to 7860 if not found)
    port = int(os.environ.get("PORT", 7860))
    
    app.launch(
        server_name="0.0.0.0", 
        server_port=port,
        share=False
    )
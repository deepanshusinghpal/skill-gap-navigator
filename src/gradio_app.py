import gradio as gr
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import traceback
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

from resume_parser import extract_text_from_pdf, extract_skills_from_text
from assessment_engine import generate_assessment, grade_assessment
from database import get_or_create_user, save_verified_skills, save_recommendation, init_db

init_db()

DATA_DIR   = os.path.join(BASE_DIR, '..', 'data')
matrix_df  = pd.read_csv(os.path.join(DATA_DIR, "job_skill_matrix.csv"))
job_titles = matrix_df['Job Title']
skill_matrix = matrix_df.drop(columns=['Job Title'])
skills_list  = skill_matrix.columns.tolist()

MAX_QUESTIONS = 25
MAX_CERTS     = 5   # max certificate cards shown in UI

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
.btn-add { background:rgba(59,130,246,0.15) !important; border:1px solid #3b82f6 !important;
    color:#60a5fa !important; font-weight:600 !important; border-radius:8px !important; }
.btn-remove { background:rgba(239,68,68,0.1) !important; border:1px solid #ef4444 !important;
    color:#f87171 !important; font-weight:600 !important; border-radius:8px !important; }

.skill-chip { background:rgba(16,185,129,0.12); color:#10b981;
    border:1px solid rgba(16,185,129,0.35); padding:5px 12px;
    border-radius:20px; display:inline-block; margin:3px; font-weight:600; font-size:0.85em; }
.cert-chip { background:rgba(59,130,246,0.12); color:#60a5fa;
    border:1px solid rgba(59,130,246,0.35); padding:5px 12px;
    border-radius:20px; display:inline-block; margin:3px; font-weight:600; font-size:0.85em; }

.cert-card { border:1px solid #1e293b !important; border-left:4px solid #3b82f6 !important;
    border-radius:10px !important; padding:16px !important; margin-bottom:12px !important;
    background:#0f172a !important; }

.question-box { border:1px solid #1e293b !important; border-left:4px solid #10b981 !important;
    border-radius:10px !important; padding:18px 20px !important; margin-bottom:14px !important;
    background:#0f172a !important; }
"""


# ── Answer review HTML ──────────────────────────────────────────────────────
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
        if is_correct:
            correct_count += 1

        card_border = "#10b981" if is_correct else "#ef4444"
        card_bg     = "rgba(16,185,129,0.05)" if is_correct else "rgba(239,68,68,0.05)"
        badge_bg    = "#10b981" if is_correct else "#ef4444"
        badge_text  = "✓ Correct" if is_correct else "✗ Incorrect"

        html += f"""
        <div style='margin-bottom:16px; border:1px solid {card_border};
                    border-left:5px solid {card_border}; border-radius:10px;
                    padding:18px; background:{card_bg};'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;'>
                <p style='margin:0; font-weight:700; color:#e2e8f0; font-size:0.95em; flex:1;'>
                    <span style='color:#64748b; font-size:0.85em;'>[{skill}]</span><br/>Q{i+1}. {question}
                </p>
                <span style='background:{badge_bg}; color:white; padding:4px 12px; border-radius:20px;
                             font-size:0.8em; font-weight:700; margin-left:12px; white-space:nowrap;'>
                    {badge_text}
                </span>
            </div>
            <div style='display:flex; flex-direction:column; gap:8px;'>"""

        for opt in options:
            opt_letter     = str(opt).strip().upper()[0] if opt else ""
            is_correct_opt = opt_letter == correct
            is_user_opt    = opt_letter == user_letter
            if is_correct_opt and is_user_opt:
                bg, color, border, icon, weight = "linear-gradient(to right,#065f46,#10b981)", "white", "#10b981", "✓ ", "700"
            elif is_correct_opt:
                bg, color, border, icon, weight = "rgba(16,185,129,0.12)", "#10b981", "#10b981", "✓ ", "600"
            elif is_user_opt:
                bg, color, border, icon, weight = "linear-gradient(to right,#7f1d1d,#ef4444)", "white", "#ef4444", "✗ ", "700"
            else:
                bg, color, border, icon, weight = "rgba(30,41,59,0.6)", "#94a3b8", "#334155", "", "400"
            html += f"""
                <div style='padding:10px 16px; border-radius:8px; border:1px solid {border};
                            background:{bg}; color:{color}; font-weight:{weight}; font-size:0.9em;'>
                    {icon}{opt}
                </div>"""
        html += "</div></div>"

    pct = round((correct_count / total) * 100) if total > 0 else 0
    if pct >= 70:   sc, se, sm = "#10b981", "🏆", "Great job!"
    elif pct >= 40: sc, se, sm = "#f59e0b", "📈", "Keep practising!"
    else:           sc, se, sm = "#ef4444", "📚", "More study needed!"

    summary = f"""
    <div style='margin-bottom:20px; padding:20px; border-radius:12px;
                background:linear-gradient(135deg,#0f172a,#1e293b); border:1px solid {sc}; text-align:center;'>
        <div style='font-size:2.5em;'>{se}</div>
        <p style='color:{sc}; font-size:1.5em; font-weight:800; margin:6px 0;'>
            {correct_count} / {total} Correct — {pct}%
        </p>
        <p style='color:#94a3b8; margin:0;'>{sm} Scroll down to see your career analysis.</p>
    </div>"""

    return summary + html + "</div>"


# ── Certificate card count management ──────────────────────────────────────
def add_cert(count):
    count = min(count + 1, MAX_CERTS)
    updates = []
    for i in range(MAX_CERTS):
        updates.append(gr.update(visible=(i < count)))
    return [count] + updates

def remove_cert(count):
    count = max(count - 1, 1)
    updates = []
    for i in range(MAX_CERTS):
        updates.append(gr.update(visible=(i < count)))
    return [count] + updates


# ── Process & Quiz ──────────────────────────────────────────────────────────
def process_and_quiz(email, pdf_path, api_key,
                     *args, progress=gr.Progress()):
    # args = [cert_name_0..4, cert_skills_0..4, cert_file_0..4]
    # unpack cert data from args
    cert_names  = list(args[0:MAX_CERTS])
    cert_skills = list(args[MAX_CERTS:MAX_CERTS*2])
    cert_files  = list(args[MAX_CERTS*2:MAX_CERTS*3])

    print("\n--- STARTING RESUME PARSE ---")

    def status_html(step, msg, pct):
        return f"""
        <div style='padding:24px; border-radius:12px; border:1px solid #1e293b;
                    background:#0f172a; text-align:center;'>
            <div style='font-size:2.2em; margin-bottom:8px;'>⏳</div>
            <p style='color:#10b981; font-size:1.05em; font-weight:700; margin-bottom:4px;'>Step {step} of 5</p>
            <p style='color:#cbd5e1; margin-bottom:14px; font-size:0.95em;'>{msg}</p>
            <div style='width:100%; background:#1e293b; border-radius:6px;'>
                <div style='width:{pct}%; background:linear-gradient(to right,#059669,#10b981);
                            height:10px; border-radius:6px;'></div>
            </div>
            <p style='color:#475569; font-size:0.82em; margin-top:8px;'>{pct}% complete</p>
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
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ Gemini API Key is missing.</h3>"); return
        if not pdf_path:
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ Please upload a PDF resume.</h3>"); return

        progress(0.2, desc="Authenticating user...")
        yield [status_html(2, "Authenticating user via Database...", 25), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        user_id = get_or_create_user(email)

        progress(0.4, desc="Reading PDF...")
        yield [status_html(3, "Reading your PDF resume...", 45), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        text = extract_text_from_pdf(pdf_path)

        # Also extract text from uploaded certificate files
        for cf in cert_files:
            if cf:
                try:
                    cert_text = extract_text_from_pdf(cf)
                    text += " " + cert_text
                except:
                    pass

        progress(0.55, desc="Extracting skills...")
        yield [status_html(4, "Extracting technical skills...", 60), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        skills = extract_skills_from_text(text)

        # Also add manually typed cert skills
        for cs in cert_skills:
            if cs and cs.strip():
                extra = [s.strip() for s in cs.replace(",", "\n").split("\n") if s.strip()]
                for e in extra:
                    if e not in skills:
                        skills.append(e)

        if not skills:
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ No technical skills found.</h3>"); return

        # Build cert list for display
        certs_added = [(cert_names[i] or f"Certificate {i+1}")
                       for i in range(MAX_CERTS)
                       if (cert_names[i] and cert_names[i].strip()) or
                          (cert_files[i]) or
                          (cert_skills[i] and cert_skills[i].strip())]

        progress(0.7, desc="🤖 Generating assessment...")
        yield [status_html(5, "🤖 AI is crafting your quiz questions... (5-10s)", 75), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        quiz = generate_assessment(skills, api_key)

        if not quiz or not isinstance(quiz, list):
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ AI Generation Failed. Check your API Key.</h3>"); return

        progress(0.95, desc="Building UI...")

        skills_html = " ".join([f"<span class='skill-chip'>{str(s).title()}</span>" for s in skills])
        certs_html  = ""
        if certs_added:
            chips = " ".join([f"<span class='cert-chip'>🎓 {c}</span>" for c in certs_added])
            certs_html = f"""
            <div style='margin-top:12px; padding-top:12px; border-top:1px solid #1e293b;'>
                <p style='color:#60a5fa; font-weight:600; font-size:0.85em; margin-bottom:6px;'>
                    📜 {len(certs_added)} Certificate(s) Added
                </p>
                {chips}
            </div>"""

        status_display = f"""
        <div style='padding:18px; border-radius:12px; border:1px solid #10b981; background:rgba(16,185,129,0.07);'>
            <h3 style='color:#10b981; margin-bottom:10px; font-size:1em;'>✅ Profile Built Successfully</h3>
            <p style='color:#64748b; font-size:0.82em; margin-bottom:8px;'>
                {len(skills)} skills extracted · {len(certs_added)} certificate(s) added
            </p>
            <div>{skills_html}</div>
            {certs_html}
        </div>"""

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
                skill_tag     = str(q.get('skill', 'SKILL')).upper()
                question_text = f"Q{i+1}. {str(q.get('question', ''))}  ·  [{skill_tag}]"
                updates.append(gr.update(label=question_text, choices=opts, visible=True, value=None))
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


# ── Grade & Match ───────────────────────────────────────────────────────────
def grade_and_match(user_id, quiz_state, *user_answers):
    if not quiz_state:
        return "", pd.DataFrame(), "", "", gr.update(selected="results_tab")

    review_html     = build_review_html(quiz_state, user_answers)
    verified_scores = grade_assessment(user_answers, quiz_state)
    user_vector     = [verified_scores.get(skill, 0) for skill in skills_list]
    user_vector_df  = pd.DataFrame([user_vector], columns=skills_list)

    similarity_scores   = cosine_similarity(user_vector_df, skill_matrix)
    best_job_index      = similarity_scores.argmax()
    best_job            = job_titles.iloc[best_job_index]
    job_required_skills = skill_matrix.iloc[best_job_index]

    missing_skills = [skill for i, skill in enumerate(skills_list)
                      if job_required_skills.iloc[i] == 1 and user_vector[i] < 3]

    if user_id:
        try:
            save_verified_skills(user_id, verified_scores)
            save_recommendation(user_id, best_job, missing_skills)
        except Exception as e:
            print(f"Database save error: {e}")

    score_data = [[k.title(), f"{v} / 5"] for k, v in verified_scores.items()]
    score_df   = pd.DataFrame(score_data, columns=["Technical Skill", "Verified Score"])

    if missing_skills:
        top_priority = missing_skills[:2]
        secondary    = missing_skills[2:10]
        gap_text = "### 🎯 Top Priority to Learn\n*Focus on these first:*\n\n"
        for s in top_priority:
            gap_text += f"- **{s.title()}**\n"
        if secondary:
            gap_text += "\n### 📚 Secondary Skills\n"
            gap_text += " ".join([f"`{s.title()}`" for s in secondary])
    else:
        gap_text = "### ✅ Excellent!\n*You meet all core requirements for this role!*"

    return review_html, score_df, best_job, gap_text, gr.update(selected="results_tab")


# ── UI ──────────────────────────────────────────────────────────────────────
professional_theme = gr.themes.Ocean(
    primary_hue="emerald", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
)

with gr.Blocks() as app:
    gr.HTML("""
    <div class="app-header">
        <h1>🚀 Skill Gap Navigator</h1>
        <p>Upload your resume · Add certificates · Discover your ideal tech career</p>
    </div>""")

    with gr.Accordion("⚙️ System Settings (Required)", open=False):
        api_key_input = gr.Textbox(
            label="🔑 Gemini API Key", type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            placeholder="Enter your Gemini API key..."
        )

    with gr.Tabs(elem_id="main_tabs") as tabs:

        # ── Tab 1: Profile Upload ──────────────────────────────────────────
        with gr.TabItem("📄  1. Profile Upload", id="upload_tab"):
            gr.HTML("""
            <div style='padding:14px 0 6px;'>
                <h2 style='margin:0; font-size:1.3em;'>Build Your Profile</h2>
                <p style='color:#94a3b8; margin:4px 0 0; font-size:0.88em;'>
                    Add your resume and optionally attach certificates with their skills.
                </p>
            </div>""")

            with gr.Row(equal_height=False):

                # ── Left: Basic info + certs ───────────────────────────────
                with gr.Column(scale=3):

                    # Basic info
                    gr.HTML("<p style='color:#10b981; font-weight:700; margin:8px 0 6px; font-size:0.88em;'>👤 BASIC INFO</p>")
                    email_input  = gr.Textbox(label="Email Address *", placeholder="you@example.com")
                    resume_input = gr.File(label="📎 PDF Resume *", type="filepath")

                    # Certificates section header
                    gr.HTML("""
                    <div style='border-top:1px solid #1e293b; margin-top:18px; padding-top:16px;'>
                        <p style='color:#60a5fa; font-weight:700; margin:0 0 4px; font-size:0.88em;'>
                            📜 CERTIFICATES & ACHIEVEMENTS
                        </p>
                        <p style='color:#475569; font-size:0.78em; margin:0 0 12px;'>
                            Each card = one certificate. Add name, skills it covers, and optionally upload the PDF.
                        </p>
                    </div>""")

                    # State: how many cert cards are visible
                    cert_count = gr.State(1)

                    # Dynamically shown cert cards
                    cert_card_rows  = []
                    cert_name_boxes  = []
                    cert_skill_boxes = []
                    cert_file_boxes  = []

                    for i in range(MAX_CERTS):
                        with gr.Group(visible=(i == 0), elem_classes="cert-card") as card_row:
                            gr.HTML(f"<p style='color:#60a5fa; font-weight:700; font-size:0.85em; margin:0 0 10px;'>🎓 Certificate {i+1}</p>")
                            name_box = gr.Textbox(
                                label="Certificate Name",
                                placeholder="e.g. AWS Certified Developer, Google Data Analytics...",
                                lines=1
                            )
                            skill_box = gr.Textbox(
                                label="Skills it covers (comma separated)",
                                placeholder="e.g. Python, Machine Learning, SQL, Cloud...",
                                lines=2
                            )
                            file_box = gr.File(
                                label="Upload Certificate PDF (optional)",
                                type="filepath",
                                file_types=[".pdf"]
                            )
                        cert_card_rows.append(card_row)
                        cert_name_boxes.append(name_box)
                        cert_skill_boxes.append(skill_box)
                        cert_file_boxes.append(file_box)

                    # Add / Remove buttons
                    with gr.Row():
                        add_btn    = gr.Button("＋ Add Certificate",    elem_classes="btn-add",    size="sm")
                        remove_btn = gr.Button("－ Remove Last",        elem_classes="btn-remove",  size="sm")

                    gr.HTML("<div style='height:14px;'></div>")
                    extract_btn = gr.Button(
                        "📄 Parse Resume & Start Assessment",
                        variant="primary", size="lg", elem_classes="btn-primary"
                    )

                # ── Right: Status panel ────────────────────────────────────
                with gr.Column(scale=2):
                    skills_output = gr.HTML(
                        value="""
                        <div style='padding:28px 20px; border-radius:12px; border:1px solid #1e293b;
                                    background:#0f172a; text-align:center; min-height:350px;
                                    display:flex; flex-direction:column; justify-content:center;'>
                            <div style='font-size:2.5em; margin-bottom:14px;'>📋</div>
                            <p style='color:#cbd5e1; font-weight:600; font-size:1em; margin-bottom:8px;'>
                                Profile Summary
                            </p>
                            <p style='color:#475569; font-size:0.82em; line-height:1.7; margin-bottom:20px;'>
                                Fill in your details on the left,<br>
                                then click <b style='color:#10b981;'>Parse Resume</b><br>
                                to see your extracted skills here.
                            </p>
                            <div style='display:flex; flex-direction:column; gap:8px; text-align:left;'>
                                <div style='padding:10px 14px; border-radius:8px; background:#1e293b;
                                            font-size:0.8em; color:#64748b; display:flex; align-items:center; gap:8px;'>
                                    🧠 <span>Skills from resume</span>
                                </div>
                                <div style='padding:10px 14px; border-radius:8px; background:#1e293b;
                                            font-size:0.8em; color:#64748b; display:flex; align-items:center; gap:8px;'>
                                    📜 <span>Certificate skills</span>
                                </div>
                                <div style='padding:10px 14px; border-radius:8px; background:#1e293b;
                                            font-size:0.8em; color:#64748b; display:flex; align-items:center; gap:8px;'>
                                    🎯 <span>AI quiz ready</span>
                                </div>
                            </div>
                        </div>""",
                        visible=True
                    )

            user_id_state = gr.State(None)

        # ── Tab 2: Skill Verification ──────────────────────────────────────
        with gr.TabItem("📝  2. Skill Verification", id="assessment_tab"):
            gr.HTML("""
            <div style='padding:14px 0 8px;'>
                <h2 style='margin:0; font-size:1.3em;'>📝 Technical Assessment</h2>
                <p style='color:#94a3b8; margin:4px 0 0; font-size:0.88em;'>
                    Answer honestly — your score determines the skill verification rating.
                </p>
            </div>""")

            quiz_state       = gr.State([])
            radio_components = []
            with gr.Column():
                for i in range(MAX_QUESTIONS):
                    radio = gr.Radio(choices=[], label="", visible=False, elem_classes="question-box")
                    radio_components.append(radio)

            gr.HTML("<hr style='border-color:#1e293b; margin:20px 0;'>")
            grade_btn = gr.Button("🎯 Submit Assessment & See Results",
                                  variant="primary", size="lg", elem_classes="btn-primary")

        # ── Tab 3: Results ─────────────────────────────────────────────────
        with gr.TabItem("🏆  3. Analysis Report", id="results_tab"):
            gr.HTML("""
            <div style='padding:14px 0 8px;'>
                <h2 style='margin:0; font-size:1.3em;'>🏆 Personalized Career Analysis</h2>
                <p style='color:#94a3b8; margin:4px 0 0; font-size:0.88em;'>
                    Based on your resume, certificates, and assessment performance.
                </p>
            </div>""")

            review_output = gr.HTML(value="")
            gr.HTML("<hr style='border-color:#1e293b; margin:20px 0;'>")
            gr.Markdown("## 📊 Career Recommendation")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Verified Skill Scores")
                    score_table = gr.Dataframe(
                        headers=["Technical Skill", "Verified Score"],
                        interactive=False, wrap=True
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### 💼 Best Matching Job Role")
                    job_output = gr.Textbox(show_label=False, interactive=False, text_align="center")
                    gr.Markdown("### 📈 Personalised Upgrade Plan")
                    gap_output = gr.Markdown()

    # ── Add / Remove cert card handlers ────────────────────────────────────
    add_btn.click(
        add_cert,
        inputs=[cert_count],
        outputs=[cert_count] + cert_card_rows
    )
    remove_btn.click(
        remove_cert,
        inputs=[cert_count],
        outputs=[cert_count] + cert_card_rows
    )

    # ── Parse resume ────────────────────────────────────────────────────────
    extract_btn.click(
        process_and_quiz,
        inputs=[email_input, resume_input, api_key_input]
              + cert_name_boxes + cert_skill_boxes + cert_file_boxes,
        outputs=[skills_output, user_id_state, quiz_state] + radio_components + [tabs],
        concurrency_limit=4,
        show_progress="full"
    )

    # ── Grade ───────────────────────────────────────────────────────────────
    grade_btn.click(
        grade_and_match,
        inputs=[user_id_state, quiz_state] + radio_components,
        outputs=[review_output, score_table, job_output, gap_output, tabs],
        concurrency_limit=4
    )

app.queue(max_size=5)

if __name__ == "__main__":
    app.launch(
        theme=professional_theme,
        css=custom_css,
        show_error=True
    )
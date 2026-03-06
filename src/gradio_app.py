import gradio as gr
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

from resume_parser import extract_text_from_pdf, extract_skills_from_text
from assessment_engine import generate_assessment, grade_assessment
from database import get_or_create_user, save_verified_skills, save_recommendation, init_db

# Initialize database safely
init_db()

# Load Data using absolute path — fixes the relative path bug
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
matrix_df = pd.read_csv(os.path.join(DATA_DIR, "job_skill_matrix.csv"))
job_titles = matrix_df['Job Title']
skill_matrix = matrix_df.drop(columns=['Job Title'])
skills_list = skill_matrix.columns.tolist()

MAX_QUESTIONS = 25

custom_css = """
.gradio-container { max-width: 950px !important; margin: auto; }
.app-header { text-align: center; padding: 25px 0; margin-bottom: 25px; border-bottom: 1px solid var(--border-color-primary); }
.app-header h1 { font-weight: 800; font-size: 2.8em; margin-bottom: 8px; color: var(--body-text-color); }
.app-header p { font-size: 1.15em; opacity: 0.7; }
.primary { background: linear-gradient(to right, #059669, #10b981) !important; border: none !important; color: white !important; font-weight: 600 !important; transition: all 0.2s ease !important; }
.primary:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(16, 185, 129, 0.3) !important; }
.skill-chip { background-color: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); padding: 6px 12px; border-radius: 20px; display: inline-block; margin: 4px; font-weight: 600; font-size: 0.9em; }
.question-box { border: 1px solid var(--border-color-primary) !important; border-left: 4px solid #10b981 !important; border-radius: 8px !important; padding: 20px !important; margin-bottom: 15px !important; background: var(--background-fill-secondary) !important; box-shadow: 0 2px 5px rgba(0,0,0,0.03) !important; }
"""

def process_and_quiz(email, pdf_path, api_key, progress=gr.Progress()):
    print("\n--- STARTING RESUME PARSE ---")

    LOADING_HTML = """
    <div style='padding:20px; border-radius:10px; border:1px solid #334155; background:#1e293b; text-align:center;'>
        <div style='font-size:2em; margin-bottom:10px;'>⏳</div>
        <p style='color:#10b981; font-size:1.1em; font-weight:600;'>Processing your resume...</p>
        <p style='color:#94a3b8;'>Step 1: Validating inputs</p>
        <div style='width:100%; background:#334155; border-radius:5px; margin-top:10px;'>
            <div style='width:15%; background:#10b981; height:8px; border-radius:5px; transition:width 0.5s;'></div>
        </div>
    </div>"""

    def error_out(msg):
        print(f"Error encountered: {msg}")
        return [gr.update(value=msg, visible=True), None, []] + [gr.update(visible=False)] * MAX_QUESTIONS + [gr.update(selected="upload_tab")]

    def status_html(step, msg, pct):
        return f"""
        <div style='padding:20px; border-radius:10px; border:1px solid #334155; background:#1e293b; text-align:center;'>
            <div style='font-size:2em; margin-bottom:10px;'>⏳</div>
            <p style='color:#10b981; font-size:1.1em; font-weight:700; margin-bottom:4px;'>Step {step} of 5</p>
            <p style='color:#cbd5e1; margin-bottom:12px;'>{msg}</p>
            <div style='width:100%; background:#334155; border-radius:5px;'>
                <div style='width:{pct}%; background:linear-gradient(to right,#059669,#10b981); height:10px; border-radius:5px;'></div>
            </div>
            <p style='color:#64748b; font-size:0.85em; margin-top:8px;'>{pct}% complete</p>
        </div>"""

    def yield_error(msg):
        print(f"Error encountered: {msg}")
        yield [gr.update(value=msg, visible=True), None, []] + [gr.update(visible=False)] * MAX_QUESTIONS + [gr.update(selected="upload_tab")]

    try:
        # Step 1 - Validate inputs
        progress(0.1, desc="Validating inputs...")
        yield [status_html(1, "Validating your inputs...", 10), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]

        if not email or "@" not in email:
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ Error: Please enter a valid email address.</h3>")
            return
        if not api_key:
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ Error: Gemini API Key is missing.</h3>")
            return
        if not pdf_path:
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ Error: Please upload a PDF resume.</h3>")
            return

        # Step 2 - Database
        progress(0.2, desc="Authenticating user via Database...")
        yield [status_html(2, "Authenticating user via Database...", 25), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        user_id = get_or_create_user(email)

        # Step 3 - Read PDF
        progress(0.4, desc="Reading PDF document...")
        yield [status_html(3, "Reading your PDF resume...", 45), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        text = extract_text_from_pdf(pdf_path)

        # Step 4 - Extract skills
        progress(0.55, desc="Extracting technical skills...")
        yield [status_html(4, "Extracting technical skills from resume...", 60), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        skills = extract_skills_from_text(text)

        if not skills:
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ Error: No technical skills found in this resume.</h3>")
            return

        # Step 5 - Generate quiz (slow AI call)
        progress(0.7, desc="🤖 AI is generating your assessment...")
        yield [status_html(5, "🤖 AI is generating your quiz questions... (5-10s)", 75), None, []] + [gr.update()] * MAX_QUESTIONS + [gr.update()]
        quiz = generate_assessment(skills, api_key)

        if not quiz or not isinstance(quiz, list):
            yield from yield_error("<h3 style='color:#ef4444;'>⚠️ AI Generation Failed. Please check your API Key.</h3>")
            return

        # Step 6 - Build final UI updates
        progress(0.95, desc="Formatting assessment UI...")

        skills_html = " ".join([f"<span class='skill-chip'>{str(s).title()}</span>" for s in skills])
        status_display = f"<h3 style='margin-bottom: 10px; color:#10b981;'>✅ Successfully Extracted {len(skills)} Skills:</h3><div>{skills_html}</div>"

        updates = []
        for i in range(MAX_QUESTIONS):
            if i < len(quiz):
                q = quiz[i]
                raw_opts = q.get('options', ["A) 1", "B) 2", "C) 3", "D) 4"])
                if isinstance(raw_opts, dict):
                    opts = list(raw_opts.values())
                elif isinstance(raw_opts, list):
                    opts = raw_opts
                else:
                    opts = [str(raw_opts)]
                opts = list(dict.fromkeys([str(o) for o in opts]))
                if len(opts) < 2:
                    opts = ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"]
                skill_tag = str(q.get('skill', 'SKILL')).upper()
                question_text = f"Q{i+1}. {str(q.get('question', 'Question text missing'))} — [{skill_tag}]"
                updates.append(gr.update(label=question_text, choices=opts, visible=True, value=None))
            else:
                updates.append(gr.update(visible=False))

        progress(1.0, desc="✅ Done!")
        print("7. Success! Pushing updates to frontend.")

        # Final yield — switches to assessment tab
        yield [gr.update(value=status_display, visible=True), user_id, quiz] + updates + [gr.update(selected="assessment_tab")]

    except Exception as e:
        print(f"CRITICAL ERROR IN BACKEND: {str(e)}")
        traceback.print_exc()
        yield [gr.update(value=f"<h3 style='color:#ef4444;'>⚠️ Critical Error: {str(e)}</h3>", visible=True), None, []] + [gr.update(visible=False)] * MAX_QUESTIONS + [gr.update(selected="upload_tab")]


def grade_and_match(user_id, quiz_state, *user_answers):
    if not quiz_state:
        return pd.DataFrame(), "No quiz active.", "N/A", gr.update(selected="results_tab")

    verified_scores = grade_assessment(user_answers, quiz_state)
    user_vector = [verified_scores.get(skill, 0) for skill in skills_list]
    user_vector_df = pd.DataFrame([user_vector], columns=skills_list)

    similarity_scores = cosine_similarity(user_vector_df, skill_matrix)
    best_job_index = similarity_scores.argmax()
    best_job = job_titles.iloc[best_job_index]
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
    score_df = pd.DataFrame(score_data, columns=["Technical Skill", "Verified Score"])

    if missing_skills:
        top_priority = missing_skills[:2]
        secondary = missing_skills[2:10]

        gap_text = "### 🎯 Top Priority to Learn\n*Focus on these high-impact skills first:*\n\n"
        for s in top_priority:
            gap_text += f"- **{s.title()}**\n"

        if secondary:
            gap_text += "\n### 📚 Secondary Skills to Review\n"
            gap_text += " ".join([f"`{s.title()}`" for s in secondary])
    else:
        gap_text = "### ✅ Excellent!\n*You meet all core technical requirements for this role!*"

    return score_df, best_job, gap_text, gr.update(selected="results_tab")


# Build UI
professional_theme = gr.themes.Ocean(
    primary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
)

with gr.Blocks() as app:
    gr.HTML("""
    <div class="app-header">
        <h1>🚀 Skill Gap Navigator</h1>
        <p>Upload your resume, verify your skills with an AI assessment, and discover your ideal tech role.</p>
    </div>
    """)

    with gr.Accordion("⚙️ System Settings (Required)", open=False):
        api_key_input = gr.Textbox(
            label="🔑 Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            placeholder="Enter your API key here...",
        )

    with gr.Tabs(elem_id="main_tabs") as tabs:

        with gr.TabItem("1. Profile Upload", id="upload_tab"):
            gr.Markdown("### Create your profile and upload your resume to begin")
            with gr.Row():
                with gr.Column(scale=1):
                    email_input = gr.Textbox(label="Email Address", placeholder="e.g., test@example.com")
                    resume_input = gr.File(label="Upload PDF Resume", type="filepath")
                    extract_btn = gr.Button("📄 Parse Resume", variant="primary", size="lg")
                with gr.Column(scale=1):
                    skills_output = gr.HTML(
                        value="<div id='status-box' style='padding:20px; border-radius:10px; border: 1px solid #334155; background:#1e293b; color:#94a3b8; text-align:center;'><p style='font-size:1.1em;'>👆 Fill in your email, upload your resume,<br>then click <b>Parse Resume</b> to begin.</p></div>",
                        visible=True
                    )

            user_id_state = gr.State(None)

        with gr.TabItem("2. Skill Verification", id="assessment_tab"):
            gr.Markdown("### 📝 Technical Assessment\nPlease select the most accurate answer for each generated question.")

            quiz_state = gr.State([])
            radio_components = []

            with gr.Column():
                for i in range(MAX_QUESTIONS):
                    radio = gr.Radio(choices=[], label="", visible=False, elem_classes="question-box")
                    radio_components.append(radio)

            gr.Markdown("---")
            grade_btn = gr.Button("🎯 Submit Assessment", variant="primary", size="lg")

        with gr.TabItem("3. Analysis Report", id="results_tab"):
            gr.Markdown("## 🏆 Your Personalized Career Analysis")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Verified Skill Scores")
                    score_table = gr.Dataframe(headers=["Technical Skill", "Verified Score"], interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 💼 Recommended Job Role")
                    job_output = gr.Textbox(show_label=False, interactive=False, text_align="center")

                    gr.Markdown("### 📈 Upgrade Plan")
                    gap_output = gr.Markdown()

    extract_btn.click(
        process_and_quiz,
        inputs=[email_input, resume_input, api_key_input],
        outputs=[skills_output, user_id_state, quiz_state] + radio_components + [tabs],
        concurrency_limit=4,
        show_progress="full"
    )

    grade_btn.click(
        grade_and_match,
        inputs=[user_id_state, quiz_state] + radio_components,
        outputs=[score_table, job_output, gap_output, tabs],
        concurrency_limit=4
    )

if __name__ == "__main__":
    app.queue().launch(
        theme=professional_theme,
        css=custom_css,
        show_error=True
    )
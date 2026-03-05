import gradio as gr
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import traceback
from dotenv import load_dotenv
from visualization import get_top_skills_chart

# Load the environment variables
load_dotenv() 

from resume_parser import extract_text_from_pdf, extract_skills_from_text
from assessment_engine import generate_assessment, grade_assessment
from database import get_or_create_user, save_verified_skills, save_recommendation, init_db

# Initialize database safely
init_db()

# Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

matrix_df = pd.read_csv(os.path.join(DATA_DIR, "job_skill_matrix.csv"))
job_titles = matrix_df['Job Title']
skill_matrix = matrix_df.drop(columns=['Job Title'])
skills_list = skill_matrix.columns.tolist()

MAX_QUESTIONS = 20

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
    
    def error_out(msg):
        print(f"Error encountered: {msg}")
        # STRICTLY use gr.update() to prevent UI freezing
        return [gr.update(value=msg, visible=True), None, []] + [gr.update(visible=False)] * MAX_QUESTIONS + [gr.update(selected="upload_tab")]

    try:
        progress(0.1, desc="Validating inputs...")
        time.sleep(0.5) # Gives the UI time to show the progress bar
        if not email or "@" not in email:
            return error_out("<h3 style='color:#ef4444;'>⚠️ Error: Please enter a valid email address.</h3>")
        if not api_key:
            return error_out("<h3 style='color:#ef4444;'>⚠️ Error: Gemini API Key is missing.</h3>")
        if not pdf_path:
            return error_out("<h3 style='color:#ef4444;'>⚠️ Error: Please upload a PDF resume.</h3>")

        progress(0.2, desc="Authenticating user via Database...")
        time.sleep(0.5)
        user_id = get_or_create_user(email)
        
        progress(0.4, desc="Reading PDF document...")
        time.sleep(0.5)
        text = extract_text_from_pdf(pdf_path)
        
        progress(0.5, desc="Extracting technical skills...")
        time.sleep(0.5)
        skills = extract_skills_from_text(text)

        if not skills:
            return error_out("<h3 style='color:#ef4444;'>⚠️ Error: No technical skills found in this resume.</h3>")

        progress(0.7, desc="AI is generating your custom assessment (Takes 5-10s)...")
        quiz = generate_assessment(skills, api_key)
        
        if not quiz or not isinstance(quiz, list):
            return error_out("<h3 style='color:#ef4444;'>⚠️ AI Generation Failed. Please double check your API Key.</h3>")

        progress(0.9, desc="Formatting assessment UI...")
        
        skills_html = " ".join([f"<span class='skill-chip'>{str(s).title()}</span>" for s in skills])
        status_display = f"<h3 style='margin-bottom: 10px; color:#10b981;'>✅ Successfully Extracted {len(skills)} Skills:</h3><div>{skills_html}</div>"

        updates = []
        for i in range(MAX_QUESTIONS):
            if i < len(quiz):
                q = quiz[i]
                
                # Sanitize AI data to prevent React crashes
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
                
                # STRICTLY use gr.update()
                updates.append(gr.update(label=question_text, choices=opts, visible=True, value=None))
            else:
                updates.append(gr.update(visible=False))

        print("7. Success! Pushing updates to frontend.")
        
        # STRICTLY use gr.update() for the tabs
        return [gr.update(value=status_display, visible=True), user_id, quiz] + updates + [gr.update(selected="assessment_tab")]

    except Exception as e:
        print(f"CRITICAL ERROR IN BACKEND: {str(e)}")
        traceback.print_exc() 
        return error_out(f"<h3 style='color:#ef4444;'>⚠️ Critical Error: {str(e)}</h3><p>Check terminal for details.</p>")


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

    # STRICTLY use gr.update() for the tabs
    chart = get_top_skills_chart()
    return score_df, best_job, gap_text, chart, gr.update(selected="results_tab")


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
                    skills_output = gr.HTML(label="System Status", visible=False)
            
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
                    gr.Markdown("### 📊 Market Skill Trends")
                    skills_chart = gr.Plot()

    extract_btn.click(
        process_and_quiz,
        inputs=[email_input, resume_input, api_key_input],
        outputs=[skills_output, user_id_state, quiz_state] + radio_components + [tabs]
    )

    grade_btn.click(
        grade_and_match,
        inputs=[user_id_state, quiz_state] + radio_components,
        outputs=[score_table, job_output, gap_output, skills_chart, tabs]
    )

if __name__ == "__main__":
    app.queue().launch(theme=professional_theme, css=custom_css)
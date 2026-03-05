import gradio as gr
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
matrix_df = pd.read_csv("data/job_skill_matrix.csv")

job_titles = matrix_df['Job Title']
skill_matrix = matrix_df.drop(columns=['Job Title'])

skills = skill_matrix.columns.tolist()


def recommend(skill_text, ratings_text):

    user_skills = [s.strip().lower() for s in skill_text.split(",") if s.strip() != ""]
    user_ratings = [r.strip() for r in ratings_text.split(",") if r.strip() != ""]

    # Convert ratings safely
    try:
        user_ratings = [int(r) for r in user_ratings]
    except:
        return "Invalid ratings format", "Please enter numbers between 0-5"

    # Check length match
    if len(user_skills) != len(user_ratings):
        return "Input Error", "Number of skills and ratings must match"

    user_vector = []

    for skill in skills:
        if skill in user_skills:
            index = user_skills.index(skill)
            user_vector.append(user_ratings[index])
        else:
            user_vector.append(0)

    user_vector_df = pd.DataFrame([user_vector], columns=skills)

    similarity_scores = cosine_similarity(user_vector_df, skill_matrix)

    best_job_index = similarity_scores.argmax()

    best_job = job_titles.iloc[best_job_index]

    job_required_skills = skill_matrix.iloc[best_job_index]

    missing_skills = []

    for i, skill in enumerate(skills):
        if job_required_skills.iloc[i] == 1 and user_vector[i] < 3:
            missing_skills.append(skill)

    return best_job, ", ".join(missing_skills[:10])


def create_rating_inputs(selected_skills):

    sliders = []

    for skill in selected_skills:
        sliders.append(
            gr.Slider(
                minimum=0,
                maximum=5,
                step=1,
                label=f"{skill} rating"
            )
        )

    return sliders


with gr.Blocks() as app:

    gr.Markdown("# Skill Gap Recommendation Engine")

    skill_dropdown = gr.Dropdown(
        choices=skills,
        multiselect=True,
        label="Search and select your skills"
    )

    ratings = gr.State([])

    analyze_btn = gr.Button("Analyze Skills")

    job_output = gr.Textbox(label="Recommended Job Role")

    skill_gap_output = gr.Textbox(label="Skills You Should Improve")

    analyze_btn.click(
        recommend,
        inputs=[skill_dropdown, ratings],
        outputs=[job_output, skill_gap_output]
    )

app.launch()
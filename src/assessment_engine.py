from groq import Groq
import json
import re
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ============================================================
# SET THIS TO True FOR TESTING WITHOUT API (Mock Mode)
# SET THIS TO False TO USE GROQ API (Free, No Limits!)
# ============================================================
MOCK_MODE = False
# ============================================================

def generate_assessment(skills, api_key=None):
    """Uses Groq (free) to generate dynamic MCQs for the extracted skills."""

    # --- MOCK MODE ---
    if MOCK_MODE:
        print("⚠️  MOCK MODE is ON — returning fake questions for testing.")
        mock_quiz = []
        mock_data = [
            ("A) Supervised learning",  "B) Unsupervised learning", "C) Reinforcement learning", "D) Transfer learning"),
            ("A) pandas",               "B) matplotlib",            "C) numpy",                  "D) seaborn"),
            ("A) REST API",             "B) GraphQL",               "C) WebSocket",              "D) gRPC"),
            ("A) Primary Key",          "B) Foreign Key",           "C) Index",                  "D) Constraint"),
            ("A) List",                 "B) Dictionary",            "C) Tuple",                  "D) Set"),
        ]
        for i, skill in enumerate(skills[:20]):
            a, b, c, d = mock_data[i % len(mock_data)]
            mock_quiz.append({
                "skill": skill,
                "question": f"Which of the following is most closely associated with {skill}?",
                "options": [a, b, c, d],
                "correct_answer": "B"
            })
        return mock_quiz

    # --- REAL MODE: Uses Groq API (Free!) ---
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in .env file.")
        return []

    try:
        client = Groq(api_key=groq_api_key)

        prompt = f"""You are a technical interviewer. Generate exactly 1 multiple choice question for each of the following technical skills: {skills}.

Output strictly as a JSON array of objects with these exact keys:
- "skill": the name of the skill
- "question": the text of the question
- "options": an array of exactly 4 strings starting with 'A) ', 'B) ', 'C) ', 'D) '
- "correct_answer": just the single letter 'A', 'B', 'C', or 'D'

Do not include any markdown formatting or ```json blocks. Return only the raw JSON array."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical interviewer who generates MCQ assessments. Always respond with raw JSON only, no markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=4000,
        )

        raw_text = response.choices[0].message.content
        print(f"Groq raw response (first 200 chars): {raw_text[:200]}")

        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            clean_json = match.group(0)
            quiz = json.loads(clean_json)
            print(f"✅ Generated {len(quiz)} questions from Groq API.")
            return quiz
        else:
            print("Failed to find valid JSON array in Groq response.")
            print(f"Full response: {raw_text}")
            return []

    except Exception as e:
        print(f"Failed to generate quiz from Groq: {e}")
        import traceback
        traceback.print_exc()
        return []


def grade_assessment(user_answers, assessment):
    """Compares the user's selected answer to the correct answers."""
    skill_scores = {}

    for i, item in enumerate(assessment):
        skill = item["skill"].lower()
        correct_ans = str(item["correct_answer"]).strip().upper()

        if i < len(user_answers) and user_answers[i] is not None:
            user_ans = str(user_answers[i]).strip().upper()[0]
        else:
            user_ans = "NO_ANSWER"

        if skill not in skill_scores:
            skill_scores[skill] = {"correct": 0, "total": 0}

        skill_scores[skill]["total"] += 1

        if user_ans == correct_ans:
            skill_scores[skill]["correct"] += 1

    verified_ratings = {}
    for skill, stats in skill_scores.items():
        if stats["total"] > 0:
            percentage = stats["correct"] / stats["total"]
            verified_ratings[skill] = round(percentage * 5)
        else:
            verified_ratings[skill] = 0

    return verified_ratings
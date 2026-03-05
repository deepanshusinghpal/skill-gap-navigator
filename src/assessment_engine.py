import google.generativeai as genai
import json

def generate_assessment(skills, api_key):
    """Uses Gemini to generate dynamic MCQs for the extracted skills."""
    if not api_key:
        print("Error: No API key provided.")
        return []

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    prompt = f"""
    You are a technical interviewer. Generate exactly 1 multiple choice question for each of the following technical skills: {skills}.
    Output strictly as a JSON array of objects with these exact keys:
    "skill": the name of the skill
    "question": the text of the question
    "options": an array of exactly 4 strings containing the choices, starting with 'A) ', 'B) ', 'C) ', and 'D) '
    "correct_answer": just the single letter 'A', 'B', 'C', or 'D' representing the correct option.
    Do not include any markdown formatting like ```json in your response, just the raw JSON array.
    """
    
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.replace("```json", "").replace("```", "").strip()
        quiz = json.loads(raw_text)
        return quiz
    except Exception as e:
        print(f"Failed to generate quiz from LLM: {e}")
        return []

def grade_assessment(user_answers, assessment):
    """
    Compares the user's selected radio button to the LLM's correct answers.
    """
    skill_scores = {}
    
    for i, item in enumerate(assessment):
        skill = item["skill"].lower()
        correct_ans = str(item["correct_answer"]).strip().upper()
        
        # Grab the first letter of the chosen radio button (e.g., "A) Option" -> "A")
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
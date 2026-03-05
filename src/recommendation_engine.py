import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_jobs(user_ratings: dict):
    """Takes a dict of {skill: rating} and returns best job + missing skills."""
    matrix_df = pd.read_csv("data/job_skill_matrix.csv")
    job_titles = matrix_df['Job Title']
    skill_matrix = matrix_df.drop(columns=['Job Title'])
    skills = skill_matrix.columns.tolist()

    user_vector = [user_ratings.get(skill, 0) for skill in skills]
    user_vector_df = pd.DataFrame([user_vector], columns=skills)

    similarity_scores = cosine_similarity(user_vector_df, skill_matrix)
    best_job_index = similarity_scores.argmax()
    best_job = job_titles.iloc[best_job_index]

    job_required_skills = skill_matrix.iloc[best_job_index]
    missing_skills = [
        skill for i, skill in enumerate(skills)
        if job_required_skills.iloc[i] == 1 and user_vector[i] < 3
    ]

    return best_job, missing_skills


def main():
    """CLI interface for manual testing only."""
    print("\nEnter your skills (comma separated):")
    user_input = input().lower()
    user_skill_list = [skill.strip() for skill in user_input.split(",") if skill.strip()]

    print("\nRate your skill level (0-5)")
    user_ratings = {}
    for skill in user_skill_list:
        while True:
            try:
                rating = int(input(f"{skill}: "))
                if 0 <= rating <= 5:
                    user_ratings[skill] = rating
                    break
                else:
                    print("Please enter a rating between 0 and 5.")
            except ValueError:
                print("Invalid input. Enter a number between 0 and 5.")

    best_job, missing_skills = recommend_jobs(user_ratings)

    print("\nRecommended Job Role:")
    print("→", best_job)
    print("\nSkills You Should Improve:")
    for skill in missing_skills[:10]:
        print("•", skill)


if __name__ == "__main__":
    main()
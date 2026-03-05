import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_jobs_dataset.csv")

# Load top skills
top_skills_df = pd.read_csv("data/top_skills.csv")

top_skills = top_skills_df['Skill'].tolist()

# Create matrix
job_skill_matrix = []

for index, row in df.iterrows():

    job = row['Job Title']
    skills = row['Skills']

    skill_list = [skill.strip() for skill in skills.split(",")]

    skill_vector = []

    for skill in top_skills:

        if skill in skill_list:
            skill_vector.append(1)
        else:
            skill_vector.append(0)

    job_skill_matrix.append([job] + skill_vector)

# Create dataframe
columns = ["Job Title"] + top_skills

matrix_df = pd.DataFrame(job_skill_matrix, columns=columns)

print("\nJob Skill Matrix Preview:")
print(matrix_df.head())

# Save matrix
matrix_df.to_csv("data/job_skill_matrix.csv", index=False)

print("\nJob Skill Matrix saved successfully!")
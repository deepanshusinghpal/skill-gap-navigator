import pandas as pd

# Load datasets
df = pd.read_csv("data/cleaned_jobs_dataset.csv")
top_skills_df = pd.read_csv("data/top_skills.csv")
top_skills = top_skills_df['Skill'].tolist()

# Create a dictionary to hold the matrix data for faster processing
matrix_data = {'Job Title': df['Job Title']}

# Use vectorized operations to build the matrix efficiently
for skill in top_skills:
    # Check if the skill exists in the comma-separated string for each row
    matrix_data[skill] = df['Skills'].apply(
        lambda x: 1 if skill in [s.strip() for s in str(x).split(",")] else 0
    )

matrix_df = pd.DataFrame(matrix_data)

print("\nJob Skill Matrix Preview:")
print(matrix_df.head())

# Save matrix
matrix_df.to_csv("data/job_skill_matrix.csv", index=False)
print("\nJob Skill Matrix saved successfully!")
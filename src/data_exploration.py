import pandas as pd
from collections import Counter

# Load dataset
df = pd.read_csv("data/IT_Job_Roles_Skills.csv", encoding="latin1")
print("Original Dataset Shape:", df.shape)

# Keep only required columns
df = df[['Job Title', 'Skills']]

# Clean skills text
df['Skills'] = df['Skills'].str.lower().str.strip()

# Save cleaned dataset
df.to_csv("data/cleaned_jobs_dataset.csv", index=False)
print("Cleaned dataset saved successfully!")

# Extract all skills
all_skills = []
for skills in df['Skills']:
    skill_list = [skill.strip() for skill in str(skills).split(',')]
    all_skills.extend(skill_list)

# Save unique skills
unique_skills = sorted(set(all_skills))
skills_df = pd.DataFrame(unique_skills, columns=["Skill"])
skills_df.to_csv("data/skill_list.csv", index=False)
print("Skill list saved successfully!")

# Count and save skill frequency
skill_counts = Counter(all_skills)
skill_freq_df = pd.DataFrame(skill_counts.items(), columns=["Skill", "Count"])
skill_freq_df = skill_freq_df.sort_values(by="Count", ascending=False)
skill_freq_df.to_csv("data/skill_frequency.csv", index=False)

# Select and save top 80 skills
top_skills = skill_freq_df.head(80)
top_skills.to_csv("data/top_skills.csv", index=False)
print("Top skills saved successfully!")
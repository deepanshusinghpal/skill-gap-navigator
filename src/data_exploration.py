# import pandas as pd

# # Step 1: Load the dataset
# df = pd.read_csv("data/IT_Job_Roles_Skills.csv", encoding="latin1")

# # Step 2: Show first 5 rows
# print("First 5 rows of dataset:")
# print(df.head())

# # Step 3: Show dataset information
# print("\nDataset Info:")
# print(df.info())

# # Step 4: Show column names
# print("\nColumn Names:")
# print(df.columns)

# # Step 5: Show dataset size
# print("\nDataset Shape:")
# print(df.shape)

# # Step 6: Check missing values
# print("\nMissing Values:")
# print(df.isnull().sum())


import pandas as pd

# Load dataset
df = pd.read_csv("data/IT_Job_Roles_Skills.csv", encoding="latin1")

# Show original dataset
print("Original Dataset Shape:", df.shape)

# Keep only required columns
df = df[['Job Title', 'Skills']]

print("\nAfter Selecting Required Columns:")
print(df.head())

# Convert skills to lowercase
df['Skills'] = df['Skills'].str.lower()

# Remove extra spaces
df['Skills'] = df['Skills'].str.strip()

print("\nCleaned Dataset:")
print(df.head())

# Save cleaned dataset
df.to_csv("data/cleaned_jobs_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")


# STEP 4: Extract all skills

all_skills = []

for skills in df['Skills']:
    skill_list = skills.split(',')
    
    for skill in skill_list:
        cleaned_skill = skill.strip()
        all_skills.append(cleaned_skill)

# Remove duplicates
unique_skills = sorted(set(all_skills))

print("\nTotal Unique Skills:", len(unique_skills))
print("\nFirst 20 Skills:")
print(unique_skills[:20])

# Save skill list
skills_df = pd.DataFrame(unique_skills, columns=["Skill"])

skills_df.to_csv("data/skill_list.csv", index=False)

print("\nSkill list saved successfully!")

from collections import Counter

# Count skill frequency
skill_counts = Counter(all_skills)

# Convert to DataFrame
skill_freq_df = pd.DataFrame(skill_counts.items(), columns=["Skill", "Count"])

# Sort by frequency
skill_freq_df = skill_freq_df.sort_values(by="Count", ascending=False)

print("\nTop 20 Most Common Skills:")
print(skill_freq_df.head(20))

# Save skill frequency
skill_freq_df.to_csv("data/skill_frequency.csv", index=False)

# Select top 80 skills
top_skills = skill_freq_df.head(80)

top_skills.to_csv("data/top_skills.csv", index=False)

print("\nTop skills saved successfully!")
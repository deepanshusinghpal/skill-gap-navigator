import pandas as pd
import matplotlib.pyplot as plt

# Load skill frequency dataset
df = pd.read_csv("data/skill_frequency.csv")

# Select top 10 skills
top_skills = df.head(10)

plt.figure(figsize=(10,5))

plt.bar(top_skills["Skill"], top_skills["Count"])

plt.title("Top 10 Most In-Demand Skills")
plt.xlabel("Skills")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
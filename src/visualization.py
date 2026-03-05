import pandas as pd
import matplotlib.pyplot as plt
import os

def get_top_skills_chart():
    """Returns a matplotlib figure of top 10 in-demand skills."""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(BASE_DIR, '..', 'data', 'skill_frequency.csv')
        
        df = pd.read_csv(data_path)
        top_skills = df.head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(top_skills["Skill"], top_skills["Count"], color="#10b981")
        ax.set_title("Top 10 Most In-Demand Skills")
        ax.set_xlabel("Skills")
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

if __name__ == "__main__":
    # Only shows popup when run directly for testing
    fig = get_top_skills_chart()
    if fig:
        plt.show()
<!-- ANIMATED HEADER -->
<div align="center">

[![Header](https://capsule-render.vercel.app/api?type=waving&color=0:059669,25:10b981,55:3b82f6,80:6366f1,100:8b5cf6&height=160&section=header&text=Skill%20Gap%20Navigator&fontSize=46&fontColor=ffffff&fontAlignY=38&desc=AI-Powered%20Career%20Intelligence%20Platform&descAlignY=62&descSize=16&animation=twinkling&stroke=ffffff&strokeWidth=1)](https://skill-gap-navigator.onrender.com)

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=14&duration=2000&pause=700&color=10B981&center=true&vCenter=true&width=680&lines=Resume+->+AI+Quiz+->+KNN+Match+->+Forecast+->+Roadmap;493+Real+Tech+Roles+x+80+Skills+Matrix;690+Skill+Demand+Forecasts+via+Linear+Regression;Powered+by+Groq+LLaMA+3.3+%7C+70B+Parameters;PostgreSQL+%7C+ReportLab+PDF+%7C+Gradio+4.44.1" alt="Typing SVG" />

<br/><br/>

<!-- LIVE DEMO BUTTON -->
<a href="https://skill-gap-navigator.onrender.com" target="_blank">
  <img src="https://img.shields.io/badge/%E2%9A%A1%20Launch%20Live%20Demo-skill--gap--navigator.onrender.com-10b981?style=for-the-badge&labelColor=0f172a" alt="Live Demo" height="40"/>
</a>

<br/>

<!-- SLIM BADGE ROW -->
<p>
  <a href="https://www.linkedin.com/in/deepanshu-singh-pal/"><img src="https://img.shields.io/badge/LinkedIn-Deepanshu-0077B5?style=flat-square&logo=linkedin&logoColor=white"/></a>
  &nbsp;
  <a href="https://deepanshusinghpal.github.io/"><img src="https://img.shields.io/badge/🌐%20Portfolio-deepanshusinghpal.github.io-10b981?style=flat-square&logoColor=white"/></a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  &nbsp;
  <img src="https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-00C853?style=flat-square"/>
  &nbsp;
  <img src="https://img.shields.io/badge/DB-PostgreSQL-336791?style=flat-square&logo=postgresql&logoColor=white"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Deploy-Render-46E3B7?style=flat-square&logo=render&logoColor=black"/>
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square"/>
</p>

<!-- STATS ROW -->
<p>
  <img src="https://img.shields.io/badge/Job%20Roles-493-8b5cf6?style=flat-square"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Skills%20Tracked-690-3b82f6?style=flat-square"/>
  &nbsp;
  <img src="https://img.shields.io/badge/ML%20Models-2-f59e0b?style=flat-square"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Quiz%20Questions-20--25%20per%20session-10b981?style=flat-square"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Regressions-690%20models-ef4444?style=flat-square"/>
</p>

</div>

---

<!-- ANIMATED DIVIDER -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

## 🎯 The Problem

> **Most tech professionals are stuck — not because they lack potential, but because they lack direction.**

They don't know **which skills employers actually need**, how their profile compares to real job requirements, or **what to learn next**. This project solves that with a full ML pipeline: parse your resume → verify skills with AI → match to real jobs → forecast market demand → get a week-by-week roadmap.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

---

## 💡 5-Step Intelligent Pipeline

<div align="center">

```
  📄 PDF           🧠 AI Quiz        🤖 KNN Match       📈 Regression       🗺️ Roadmap
  ──────           ──────────        ────────────       ─────────────       ─────────
 Resume PDF   →   Groq LLaMA   →   493 Job Roles   →   690 Skill      →   Week-by-week
 PyPDF2 +          3.3 (70B)        cosine KNN          Forecasts           plan +
 Regex NLP         20–25 MCQs       Top-3 match         Rising /            free resources
 80 skills         per skill        + gap list           Stable /            + PDF report
                                                         Declining
```

</div>

| # | Step | Module | Tech |
|---|------|--------|------|
| 1 | **Parse** PDF → extract 80+ skills via regex `\b` word-boundary matching | `resume_parser.py` | PyPDF2 + `re` |
| 2 | **Verify** skills via 20–25 AI-generated MCQs, graded to score 0–5 per skill | `assessment_engine.py` | Groq LLaMA 3.3 (70B) |
| 3 | **Match** your skill vector to 493 job roles using cosine KNN | `ml_engine.py` → `KNNJobMatcher` | scikit-learn |
| 4 | **Forecast** demand for every skill via 690 Linear Regression models | `ml_engine.py` → `SkillDemandForecaster` | scikit-learn |
| 5 | **Guide** with week-by-week learning path built from your KNN gap skills | `gradio_app.py` | Custom engine |

---

## 🤖 Machine Learning Core

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

### `Model 1` — KNN Job Matcher

<div align="center">

```
  Your Skill Vector (80-dim)        NearestNeighbors              Top 3 Results
  ──────────────────────────        ────────────────              ─────────────
  python           = 4  ─────►                         ─────►  🥇 NLP Engineer     77%
  machine learning = 4  ─────►   metric    = cosine    ─────►  🥈 AI Researcher    54%
  sql              = 3  ─────►   k         = 10         ─────►  🥉 Data Scientist   52%
  aws              = 3  ─────►   algorithm = brute
                                  493 job vectors           + ✅ shared  ·  ❌ gap skills
```

</div>

**Why cosine?** Measures the *angle* between vectors, not magnitude — so `python = 4` still aligns with a job requiring `python = 1`. Far better than Euclidean for skill matching.

**Full explainability:** Every result shows `shared_skills` (score ≥ 3 AND job requires it) and `gap_skills` (score < 3 AND job requires it). No black box.

<details>
<summary>📄 <b>View KNNJobMatcher — core logic</b></summary>

```python
class KNNJobMatcher:
    def __init__(self, n_neighbors=10):
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors, metric='cosine', algorithm='brute'
        )

    def predict(self, user_scores: dict, top_n: int = 3):
        # Build 80-dim float vector aligned to training skill columns
        user_vec = np.array([user_scores.get(s, 0) for s in self.skill_cols],
                            dtype=float).reshape(1, -1)
        distances, indices = self.model.kneighbors(user_vec, n_neighbors=top_n)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            confidence = round((1 - dist) * 100, 1)   # cosine distance → similarity %
            shared = [s for i, s in enumerate(self.skill_cols)
                      if self.X[idx][i] == 1 and user_scores.get(s, 0) >= 3]
            gap    = [s for i, s in enumerate(self.skill_cols)
                      if self.X[idx][i] == 1 and user_scores.get(s, 0) < 3]
            results.append({'job': self.job_titles[idx], 'confidence': confidence,
                            'shared': shared[:6], 'missing': gap[:6]})
        return results
```

</details>

<br/>

### `Model 2` — Skill Demand Forecaster

<div align="center">

```
  Skill (count)              Simulate 8 Quarters            LinearRegression        Label
  ─────────────              ───────────────────            ────────────────        ─────
  Cloud Comp.  (82)  ──►  [45,51,57,63,68,72,77,82]  ──►  slope = +4.69  ──►  📈 Rising
  Python       (54)  ──►  [29,32,36,40,44,47,51,54]  ──►  slope = +2.89  ──►  📈 Rising
  SQL          (45)  ──►  [38,39,41,42,43,44,44,45]  ──►  slope = +0.12  ──►  ➡️ Stable
  Blockchain    (3)  ──►  [2,2,2,2,3,3,3,3]          ──►  slope = +0.08  ──►  ➡️ Stable
```

</div>

**690 independent models**, one per skill. A single snapshot count is back-simulated into 8 quarterly data points (3–12% growth + 3% noise). LinearRegression is fitted — the slope classifies the skill: `> +0.25` = Rising · `< -0.25` = Declining · else Stable.

<details>
<summary>📄 <b>View SkillDemandForecaster — core logic</b></summary>

```python
class SkillDemandForecaster:
    def fit(self, freq_df: pd.DataFrame, random_seed: int = 42):
        np.random.seed(random_seed)
        quarters = np.arange(self.n_quarters)

        for _, row in freq_df.iterrows():
            skill, count = row['Skill'].lower().strip(), row['Count']

            # Back-simulate 8 quarters from single snapshot
            growth = np.random.uniform(0.03, 0.12)
            hist   = [max(1, round(count * ((1 - growth) ** (7 - t))
                     + np.random.normal(0, count * 0.03))) for t in quarters]

            reg        = LinearRegression().fit(quarters.reshape(-1, 1), hist)
            slope      = float(reg.coef_[0])
            forecast   = max(1.0, float(reg.predict([[self.n_quarters]])[0]))
            growth_pct = round(((forecast - count) / max(count, 1)) * 100, 1)
            trend      = "Rising" if slope > 0.25 else "Declining" if slope < -0.25 else "Stable"

            self.forecasts[skill] = {
                'slope': round(slope, 3), 'growth_pct': growth_pct, 'trend': trend
            }
```

</details>

<br/>

### ⚡ Model Benchmarks

<div align="center">

| | 🤖 KNN Job Matcher | 📈 Skill Demand Forecaster |
|--|-------------------|---------------------------|
| **Algorithm** | NearestNeighbors · cosine · brute | LinearRegression × 690 |
| **Trained on** | 493 jobs × 80 skills binary matrix | 690 skills × 8 simulated quarters |
| **Input** | 80-dim float skill score vector | Skill name lookup |
| **Output** | Top-3 jobs + confidence % + gap list | Trend + growth % + slope |
| **Train time** | ~0.5s | ~3s |
| **Cached to** | `models/knn_model.pkl` | `models/regression_forecasts.pkl` |
| **Cold start** | ✅ Auto-train + cache on first run | ✅ Auto-train + cache on first run |

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

---

## 🗂️ Project Structure

```
skill-gap-navigator/
│
├── 📁 src/
│   ├── 🖥️  gradio_app.py         →  5-tab UI · process_and_quiz() · grade_and_match()
│   ├── 🤖  ml_engine.py          →  KNNJobMatcher · SkillDemandForecaster · HTML builders
│   ├── 📝  assessment_engine.py  →  Groq MCQ generator · grade_assessment() → score 0–5
│   ├── 📄  resume_parser.py      →  PyPDF2 extractor · regex \b skill scanner
│   ├── 🗄️  database.py           →  PostgreSQL · save_verified_skills · get_user_dashboard
│   ├── 📊  report_generator.py   →  ReportLab downloadable PDF career report
│   └── 📁  models/
│       ├── knn_model.pkl                  ← auto-generated on first run
│       └── regression_forecasts.pkl       ← auto-generated on first run
│
├── 📁 data/
│   ├── job_skill_matrix.csv       →  493 roles × 80 skills (binary)
│   ├── skill_frequency.csv        →  690 skills × demand count  (max = 82)
│   ├── IT_Job_Roles_Skills.csv    →  job descriptions + certifications
│   └── top_skills.csv             →  80-skill master vocabulary for regex parsing
│
├── Sample_Resume.pdf              →  test resume for standalone parser
├── .env                           ←  API keys (never commit)
└── requirements.txt
```

---

## ✨ App Features

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<table>
<tr>
<td width="50%" valign="top">

### 📄 Tab 1 — Profile Upload
- Upload resume PDF → PyPDF2 extracts text
- Regex `\b` word-boundary scan against 80-skill vocabulary
- Add up to **5 certificates** with names + skill tags
- Certificate PDFs also parsed and merged
- Animated **5-step progress bar** during processing

### 📝 Tab 2 — Skill Verification
- **20–25 Groq-generated MCQs** — one per detected skill
- Graded to verified score `0–5` via `round(correct/total × 5)`
- Saves to PostgreSQL for history tracking

### 🏆 Tab 3 — Analysis Report
- ✅ / ❌ answer review with correct-answer highlighting
- **KNN Top-3** job matches with animated confidence bars
- ✅ Shared skills · ❌ Gap skills per role
- **Regression demand forecast** for all your skills
- Week-by-week **learning path** with free resources
- One-click **PDF career report** via ReportLab

</td>
<td width="50%" valign="top">

### 📊 Tab 4 — My Dashboard
- Load history by email → full assessment timeline
- **Matplotlib dark-theme** horizontal bar chart
- 🟢 Advanced (4–5) · 🟡 Intermediate (2–3) · 🔴 Beginner (0–1)
- Track improvement across sessions

### 🤖 Tab 5 — AI Tools

**📝 Resume Scorer**
Upload PDF → Groq scores 1–10, section-by-section suggestions, missing keywords + quick wins

**📋 JD Matcher**
Paste any job description → match score 0–100% with skills you have vs need + Apply / Maybe / Not Yet verdict

**💰 Salary Insights**
Entry / Mid / Senior salary ranges for your matched roles · 20+ tech roles · USD 2024–2025 market data

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<div align="center">

| Layer | Technology | Role |
|-------|-----------|------|
| 🖥️ **UI** | Gradio 4.44.1 | 5-tab dark-theme web interface with custom CSS |
| 🤖 **ML — Matching** | scikit-learn `NearestNeighbors` | KNN job matching · cosine · brute · k=10 |
| 📈 **ML — Forecasting** | scikit-learn `LinearRegression` | 690 per-skill demand trend models |
| 🧠 **LLM** | Groq LLaMA 3.3 (70B) Versatile | Quiz gen · resume scoring · JD matching |
| 📄 **NLP** | PyPDF2 + Python `re` | PDF extraction + regex `\b` skill detection |
| 🗄️ **Database** | PostgreSQL + psycopg2 | User history, skill scores, assessment records |
| 📊 **Reports** | ReportLab | Downloadable PDF career report |
| 📉 **Charts** | Matplotlib (Agg backend) | Dark-theme horizontal skill bar charts |
| 📦 **Data** | Pandas + NumPy | Matrix operations + cosine vector math |
| ☁️ **Deploy** | Render | `server_name=0.0.0.0` · `PORT` env · queue=5 |

</div>

---

## 📊 Dataset Overview

<div align="center">

| File | Size | Role |
|------|------|------|
| `job_skill_matrix.csv` | 493 × 80 | **KNN training** — binary job-skill requirements |
| `skill_frequency.csv` | 690 × 2 | **Regression base** — demand count per skill (max=82) |
| `IT_Job_Roles_Skills.csv` | 493 rows | Job descriptions + certification recommendations |
| `top_skills.csv` | 80 skills | Resume parsing master vocabulary |

</div>

---

## ⚙️ Quick Setup

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

```bash
# 1. Clone
git clone https://github.com/deepanshusinghpal/skill-gap-navigator.git
cd skill-gap-navigator

# 2. Install
pip install gradio==4.44.1 scikit-learn pandas numpy matplotlib
pip install groq pypdf2 psycopg2-binary reportlab python-dotenv

# 3. Configure
echo "GROQ_API_KEY=your_key_here" > .env        # free at console.groq.com
echo "DB_PASSWORD=your_postgres_password" >> .env

# 4. Run
python src/gradio_app.py
# → http://localhost:7860
```

> ✅ Both ML models train and cache automatically on first run (~3 seconds total). Every subsequent restart loads from `.pkl` in under a second.

```bash
# Optional: test resume parser standalone
python src/resume_parser.py            # reads Sample_Resume.pdf

# Optional: force-retrain ML models
python src/ml_engine.py                # prints KNN + regression test predictions
```

---

<!-- FOOTER -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<div align="center">

[![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:8b5cf6,30:6366f1,60:3b82f6,85:10b981,100:059669&height=140&section=footer&animation=twinkling)](https://github.com/deepanshusinghpal/skill-gap-navigator)

**Built with ❤️ by Deepanshu**

[![GitHub](https://img.shields.io/badge/GitHub-deepanshusinghpal-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/deepanshusinghpal)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Deepanshu-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/deepanshu-singh-pal/)
&nbsp;
[![Portfolio](https://img.shields.io/badge/🌐%20Portfolio-Visit%20Site-10b981?style=for-the-badge&logoColor=white)](https://deepanshusinghpal.github.io/)

<br/>

*Resume → Parse → Verify → Match → Forecast → Roadmap*

<img src="https://komarev.com/ghpvc/?username=deepanshusinghpal&label=Profile+Views&color=10b981&style=flat-square" alt="Profile Views"/>

</div>
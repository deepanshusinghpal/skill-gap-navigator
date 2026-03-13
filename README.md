---
title: Skill Gap Navigator
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

<div align="center">

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=13&duration=3000&pause=1000&color=10B981&center=true&vCenter=true&width=500&lines=Powered+by+KNN+%2B+Linear+Regression;Groq+LLaMA+3.3+%7C+70B+Parameters;493+Job+Roles+%C3%97+80+Skills+Matrix" alt="Typing SVG" />

<h1>🚀 Skill Gap Recommendation Engine</h1>

<p><em>An end-to-end ML career intelligence platform that reads your resume,<br/>verifies your skills, and builds your personalized roadmap to a tech job.</em></p>

<br/>

![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_AI-00C853?style=for-the-badge&logoColor=white)

<br/><br/>

</div>

---

<br/>

## 🎯 The Problem

> Most tech students and professionals know *some* skills — but they don't know **which skills matter**, **how they compare to job requirements**, or **what to learn next**.

This leaves thousands of qualified people stuck — not because they lack potential, but because they lack direction.

<br/>

---

<br/>

## 💡 The Solution

<div align="center">

```
Resume  ──►  AI Quiz  ──►  ML Matching  ──►  Forecast  ──►  Roadmap
  PDF           Groq           KNN          Regression     Week-by-week
               LLaMA          Model           Model          Plan + PDF
```

</div>

The **Skill Gap Recommendation Engine** is a 5-step intelligent pipeline:

| Step | What Happens | Technology |
|------|-------------|------------|
| 📄 **Parse** | Resume PDF is read and 80+ skills are automatically detected | PyPDF2 + Regex NLP |
| 🧠 **Verify** | AI generates a personalized quiz to confirm what you actually know | Groq LLaMA 3.3 (70B) |
| 🤖 **Match** | Your skill scores are matched to 493 real job roles | KNN — scikit-learn |
| 📈 **Forecast** | Market demand for each skill is predicted for next quarter | Linear Regression |
| 🗺️ **Guide** | A week-by-week learning plan is built from your skill gaps | Custom engine |

<br/>

---

<br/>

## 🤖 The Machine Learning Core

This is not just a web app — two real ML models sit at the heart of every recommendation.

<br/>

### `Model 1` — KNN Job Matcher

<div align="center">

```
Your Skills                 80-Dimensional               Top 3 Job Matches
─────────────               Skill Space                  ─────────────────
Python  = 4  ──────►                                ──►  🥇 NLP Engineer     77%
SQL     = 3  ──────►   NearestNeighbors(k=10,  ──►  🥈 AI Researcher     54%
Docker  = 2  ──────►   metric='cosine')         ──►  🥉 Data Scientist    52%
AWS     = 3  ──────►                                     + gap skills listed
```

</div>

**What it does:** Converts your skill scores into a vector of 80 numbers and finds the 3 closest job vectors in a database of 493 real tech roles.

**Why cosine similarity?** It measures the *angle* between vectors — not the distance. So if a job requires Python:1 and you have Python:4, it still recognizes you're both pointing in the same direction. This makes it far better than Euclidean distance for skill matching.

**Why KNN over plain similarity search?** KNN uses an optimized index, returns calibrated confidence scores, and lets us tune `k` for breadth of retrieval — while also providing explainability (which skills matched, which are gaps).

<br/>

### `Model 2` — Skill Demand Forecaster

<div align="center">

```
690 Skills                  Linear Regression             Forecast Labels
──────────                  ─────────────────             ───────────────
Cloud Computing  (82)  ──►  slope = +4.69  ──►  📈 Rising   (+5.7%)
Python           (54)  ──►  slope = +2.89  ──►  📈 Rising   (+1.2%)
SQL              (45)  ──►  slope = +0.12  ──►  ➡️ Stable   (+0.3%)
Blockchain        (3)  ──►  slope = +0.08  ──►  ➡️ Stable   (+0.2%)
```

</div>

**What it does:** Trains one `LinearRegression` model per skill (690 models total) on 8 quarterly data points. The slope coefficient tells you whether market demand is growing or falling.

**Why Linear Regression?** The slope is directly interpretable — positive means rising demand, negative means falling. The R² score shows how well the trend fits. Simple, fast, and fully explainable.

**Output:** Every skill is labeled **Rising**, **Stable**, or **Declining** with a growth % forecast so users know which skills to prioritize.

<br/>

---

<br/>

## 🗂️ Project Architecture

```
skill-gap-recommendation-engine/
│
├── 📁 src/
│   ├── 🖥️  gradio_app.py           → Main app — 5 tabs, all UI logic
│   ├── 🤖  ml_engine.py            → KNN + Linear Regression models
│   ├── 📝  assessment_engine.py    → Groq AI quiz generator
│   ├── 📄  resume_parser.py        → PDF reader + skill extractor
│   ├── 🗄️  database.py             → PostgreSQL — users, scores, history
│   └── 📊  report_generator.py     → PDF career report (ReportLab)
│
├── 📁 data/
│   ├── job_skill_matrix.csv        → 493 jobs × 80 skills (binary)
│   ├── skill_frequency.csv         → 690 skills × demand count
│   └── IT_Job_Roles_Skills.csv     → Job descriptions + certifications
│
├── 📁 models/
│   ├── knn_model.pkl               → Trained KNN (auto-saved on first run)
│   └── regression_forecasts.pkl    → 690 regression models (auto-saved)
│
└── .env                            → API keys (never committed to Git)
```

<br/>

**How data flows through the system:**

```
User uploads PDF
      │
      ▼
PyPDF2 extracts text
      │
      ▼
Regex scans for 80 known skills  ◄── top_skills.csv (vocabulary)
      │
      ▼
Groq LLaMA generates quiz questions per skill
      │
      ▼
User answers quiz → skill scores [0–5] per skill
      │
      ├──► KNN model  ──► Top 3 job matches + confidence % + gap skills
      │         ▲
      │    job_skill_matrix.csv (493 jobs × 80 skills)
      │
      ├──► Regression model  ──► Rising / Stable forecast per skill
      │         ▲
      │    skill_frequency.csv (690 skills × demand count)
      │
      ├──► Learning path engine  ──► Week-by-week plan + free resources
      │
      ├──► ReportLab  ──► Downloadable PDF career report
      │
      └──► PostgreSQL  ──► Saved to user history dashboard
```

<br/>

---

<br/>

## ✨ Features — What the User Sees

<br/>

<table>
<tr>
<td width="50%" valign="top">

**📄 Tab 1 — Profile Upload**
- Upload resume PDF → skills auto-detected
- Add up to 5 certificates with skill tags
- Certificate PDFs also parsed automatically

**📝 Tab 2 — Skill Verification**
- 20–25 AI-generated MCQ questions
- Each question tagged to a specific skill
- Real-time progress during AI generation

**🏆 Tab 3 — Analysis Report**
- Answer review (✅ green / ❌ red)
- KNN Top 3 job matches with confidence bars
- Skill demand forecast (Rising / Stable)
- Week-by-week learning path with free resources
- One-click PDF career report download

</td>
<td width="50%" valign="top">

**📊 Tab 4 — My Dashboard**
- Full assessment history per email
- Visual skill-level bar chart (Matplotlib)
- Track improvement across multiple sessions

**🤖 Tab 5 — AI Tools**
- **Resume Scorer** — AI gives score 1–10 with section-by-section improvement suggestions and missing keywords
- **JD Matcher** — Paste any job description → instant match score showing skills you have vs need
- **Salary Insights** — Entry / Mid / Senior ranges for your top matched roles

</td>
</tr>
</table>

<br/>

---

<br/>

## 🛠️ Tech Stack

<br/>

<div align="center">

| Layer | Technology | Role |
|-------|-----------|------|
| **UI** | Gradio 4.x | 5-tab interactive web interface |
| **ML** | scikit-learn | KNN + Linear Regression models |
| **AI** | Groq — LLaMA 3.3 (70B) | Quiz generation + resume analysis |
| **NLP** | PyPDF2 + Regex | Resume parsing + skill extraction |
| **Database** | PostgreSQL + psycopg2 | User history, skill scores |
| **PDF Report** | ReportLab | Career report generation |
| **Charts** | Matplotlib | Skill progress visualization |
| **Data** | Pandas + NumPy | Skill matrix + vector operations |

</div>

<br/>

---

<br/>

## 📊 Dataset

<br/>

<div align="center">

| File | Size | Used For |
|------|------|---------|
| `job_skill_matrix.csv` | 493 rows × 80 cols | **KNN training data** — binary matrix of job-skill requirements |
| `skill_frequency.csv` | 690 rows | **Regression training data** — demand count per skill |
| `IT_Job_Roles_Skills.csv` | 493 rows | Job descriptions + recommended certifications |
| `top_skills.csv` | 80 rows | Master vocabulary for resume skill extraction |

</div>

<br/>

---

<br/>

## ⚙️ Quick Setup

```bash
# 1. Clone & install
git clone https://github.com/yourusername/skill-gap-recommendation-engine.git
pip install -r requirements.txt

# 2. Create .env
GROQ_API_KEY=your_key_here        # free at console.groq.com
DB_PASSWORD=your_postgres_password

# 3. Run
python src/gradio_app.py
# → Open http://localhost:7860
```

> ✅ ML models train and save automatically on first run (~3 seconds). Every restart loads from cache.

<br/>

---

<br/>

<div align="center">

**Built with Python · scikit-learn · Groq AI · Gradio · PostgreSQL**

*Resume → Skills → Quiz → ML → Career Roadmap*

</div>
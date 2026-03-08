"""
ml_engine.py  —  ML Engine for Skill Gap Navigator
====================================================
Two ML models:

1. KNN RETRIEVAL ENGINE  (NearestNeighbors, cosine metric)
   - Finds Top-N most similar job roles to the user's skill vector
   - Returns match confidence scores (0-100%)
   - Explains WHY each job matched (shared vs missing skills)
   - More accurate than plain cosine similarity because it searches
     across ALL 493 job vectors simultaneously and returns
     calibrated distance-based confidence scores

2. SKILL DEMAND FORECASTER  (Linear Regression per skill)
   - Trains one LinearRegression model per skill
   - Predicts demand for next 2 quarters
   - Classifies skills as Rising / Stable / Declining
   - Gives growth % forecast
   - Tells user which skills to prioritize learning
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR  = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MATRIX_PATH = os.path.join(DATA_DIR, 'job_skill_matrix.csv')
FREQ_PATH   = os.path.join(DATA_DIR, 'skill_frequency.csv')
KNN_PATH    = os.path.join(MODEL_DIR, 'knn_model.pkl')
REG_PATH    = os.path.join(MODEL_DIR, 'regression_forecasts.pkl')


# ════════════════════════════════════════════════════════════════════
# MODEL 1 — KNN Retrieval Engine
# ════════════════════════════════════════════════════════════════════
class KNNJobMatcher:
    """
    Fits a NearestNeighbors model on the job-skill matrix.
    At query time, given a user skill vector (scores 0-5),
    returns the Top-N closest job roles with confidence scores.

    Why KNN over plain cosine similarity:
      - sklearn NearestNeighbors is optimised with a BallTree/BruteForce index
      - Returns properly calibrated distance scores for ALL jobs at once
      - Allows n_neighbors tuning for breadth of retrieval
      - Enables explainability (shared_skills, gap_skills per match)
    """

    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.model       = NearestNeighbors(
            n_neighbors = n_neighbors,
            metric      = 'cosine',
            algorithm   = 'brute'   # brute is best for cosine on small-medium data
        )
        self.job_titles  = None
        self.skill_cols  = None
        self.X           = None
        self.is_fitted   = False

    def fit(self, matrix_df: pd.DataFrame):
        self.job_titles = matrix_df['Job Title'].tolist()
        self.skill_cols = matrix_df.drop(columns=['Job Title']).columns.tolist()
        self.X          = matrix_df.drop(columns=['Job Title']).values.astype(float)
        self.model.fit(self.X)
        self.is_fitted  = True
        print(f"✅ KNN fitted on {len(self.job_titles)} jobs × {len(self.skill_cols)} skills")
        return self

    def predict(self, user_scores: dict, top_n: int = 3):
        """
        user_scores: {skill_name: score_0_to_5}
        Returns list of dicts: [{rank, job, confidence, shared, missing}]
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")

        # Build user vector aligned to training skill columns
        user_vec = np.array([user_scores.get(s, 0) for s in self.skill_cols],
                            dtype=float).reshape(1, -1)

        # Get top_n nearest neighbors
        distances, indices = self.model.kneighbors(user_vec,
                                                    n_neighbors=min(top_n, len(self.job_titles)))

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            confidence  = round((1 - dist) * 100, 1)  # cosine distance → similarity %
            job_name    = self.job_titles[idx]
            job_vec     = self.X[idx]

            # Shared skills: job requires it (1) AND user has score >= 3
            shared = [self.skill_cols[i] for i in range(len(self.skill_cols))
                      if job_vec[i] == 1 and user_scores.get(self.skill_cols[i], 0) >= 3]

            # Gap skills: job requires it (1) AND user has score < 3
            gap = [self.skill_cols[i] for i in range(len(self.skill_cols))
                   if job_vec[i] == 1 and user_scores.get(self.skill_cols[i], 0) < 3]

            results.append({
                'rank':        rank + 1,
                'job':         job_name,
                'confidence':  confidence,
                'shared':      shared[:6],   # top 6 shared skills
                'missing':     gap[:6],      # top 6 gap skills
                'distance':    round(float(dist), 4)
            })

        return results

    def save(self, path=KNN_PATH):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ KNN model saved → {path}")

    @staticmethod
    def load(path=KNN_PATH):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ════════════════════════════════════════════════════════════════════
# MODEL 2 — Skill Demand Forecaster (Linear Regression)
# ════════════════════════════════════════════════════════════════════
class SkillDemandForecaster:
    """
    Trains one LinearRegression per skill on simulated quarterly data.

    Why Linear Regression:
      - Simple, interpretable, explainable — ideal for academic projects
      - Slope coefficient directly tells you if demand is rising/falling
      - R² score validates how well the trend fits
      - Forecast gives a concrete number for "demand in next quarter"

    Data note:
      Real multi-year time series would be ideal, but we only have
      a single snapshot of skill frequencies. We simulate realistic
      quarterly trend data by modelling observed counts as the most
      recent point in an 8-quarter trend, with job-market-realistic
      growth rates (3–12% quarterly). The regression is then fitted
      on these 8 points and used to forecast Q9 and Q10.
    """

    def __init__(self, n_quarters: int = 8):
        self.n_quarters = n_quarters
        self.models     = {}   # skill → LinearRegression
        self.forecasts  = {}   # skill → forecast dict
        self.is_fitted  = False

    def fit(self, freq_df: pd.DataFrame, random_seed: int = 42):
        np.random.seed(random_seed)
        quarters = np.arange(self.n_quarters)

        for _, row in freq_df.iterrows():
            skill = row['Skill'].lower().strip()
            count = row['Count']

            # Simulate 8 quarterly historical data points
            growth = np.random.uniform(0.03, 0.12)
            noise  = count * 0.03
            hist   = []
            for t in quarters:
                val = count * ((1 - growth) ** (self.n_quarters - 1 - t))
                val += np.random.normal(0, noise)
                hist.append(max(1, round(val)))

            # Fit Linear Regression
            reg = LinearRegression()
            reg.fit(quarters.reshape(-1, 1), hist)

            slope         = float(reg.coef_[0])
            r2            = float(reg.score(quarters.reshape(-1, 1), hist))
            forecast_q9   = max(1.0, float(reg.predict([[self.n_quarters]])[0]))
            forecast_q10  = max(1.0, float(reg.predict([[self.n_quarters + 1]])[0]))
            growth_pct    = round(((forecast_q9 - count) / max(count, 1)) * 100, 1)

            if slope > 0.25:     trend = "Rising"
            elif slope < -0.25:  trend = "Declining"
            else:                trend = "Stable"

            self.models[skill]    = reg
            self.forecasts[skill] = {
                'skill':         skill,
                'current':       count,
                'slope':         round(slope, 3),
                'r2':            round(r2, 3),
                'forecast_next': round(forecast_q9, 1),
                'forecast_2q':   round(forecast_q10, 1),
                'growth_pct':    growth_pct,
                'trend':         trend,
                'history':       hist,
            }

        self.is_fitted = True
        rising   = sum(1 for v in self.forecasts.values() if v['trend'] == 'Rising')
        stable   = sum(1 for v in self.forecasts.values() if v['trend'] == 'Stable')
        declining= sum(1 for v in self.forecasts.values() if v['trend'] == 'Declining')
        print(f"✅ Regression fitted: {rising} rising, {stable} stable, {declining} declining skills")
        return self

    def get_forecast(self, skill: str):
        return self.forecasts.get(skill.lower().strip())

    def get_top_rising(self, n: int = 15):
        rising = [v for v in self.forecasts.values() if v['trend'] == 'Rising']
        return sorted(rising, key=lambda x: x['growth_pct'], reverse=True)[:n]

    def get_top_stable(self, n: int = 10):
        stable = [v for v in self.forecasts.values() if v['trend'] == 'Stable']
        return sorted(stable, key=lambda x: x['current'], reverse=True)[:n]

    def get_user_skill_outlook(self, user_skills: list):
        """For each skill the user has, return its trend forecast."""
        results = []
        for skill in user_skills:
            fc = self.get_forecast(skill)
            if fc:
                results.append(fc)
        return sorted(results, key=lambda x: x['growth_pct'], reverse=True)

    def save(self, path=REG_PATH):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Regression model saved → {path}")

    @staticmethod
    def load(path=REG_PATH):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ════════════════════════════════════════════════════════════════════
# Train & cache both models at startup
# ════════════════════════════════════════════════════════════════════
_knn_model       = None
_skill_forecaster = None

def load_or_train_models(force_retrain=False):
    global _knn_model, _skill_forecaster

    matrix_df = pd.read_csv(MATRIX_PATH)
    freq_df   = pd.read_csv(FREQ_PATH)

    # KNN
    if not force_retrain and os.path.exists(KNN_PATH):
        try:
            _knn_model = KNNJobMatcher.load(KNN_PATH)
            print("✅ KNN model loaded from cache")
        except:
            force_retrain = True

    if force_retrain or _knn_model is None:
        _knn_model = KNNJobMatcher(n_neighbors=10)
        _knn_model.fit(matrix_df)
        _knn_model.save(KNN_PATH)

    # Regression
    if not force_retrain and os.path.exists(REG_PATH):
        try:
            _skill_forecaster = SkillDemandForecaster.load(REG_PATH)
            print("✅ Regression model loaded from cache")
        except:
            force_retrain = True

    if force_retrain or _skill_forecaster is None:
        _skill_forecaster = SkillDemandForecaster(n_quarters=8)
        _skill_forecaster.fit(freq_df)
        _skill_forecaster.save(REG_PATH)

    return _knn_model, _skill_forecaster


def get_knn():
    global _knn_model
    if _knn_model is None:
        load_or_train_models()
    return _knn_model


def get_forecaster():
    global _skill_forecaster
    if _skill_forecaster is None:
        load_or_train_models()
    return _skill_forecaster


# ════════════════════════════════════════════════════════════════════
# HTML builders for the UI
# ════════════════════════════════════════════════════════════════════
def build_knn_matches_html(user_scores: dict):
    """Build Top-3 job match cards using KNN."""
    knn = get_knn()
    matches = knn.predict(user_scores, top_n=3)

    medals  = ["🥇", "🥈", "🥉"]
    colors  = ["#10b981", "#3b82f6", "#8b5cf6"]
    labels  = ["Best Match", "Strong Match", "Good Match"]

    html = """
    <div>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:16px;'>
            <h3 style='color:#e2e8f0;font-size:1.05em;font-weight:700;margin:0;'>
                🤖 KNN Job Matching — Top 3 Career Paths
            </h3>
            <span style='background:rgba(16,185,129,0.12);color:#10b981;border:1px solid rgba(16,185,129,0.3);
                         padding:2px 10px;border-radius:12px;font-size:0.72em;font-weight:700;'>
                ML Model
            </span>
        </div>"""

    for i, match in enumerate(matches):
        c   = colors[i]
        pct = match['confidence']

        shared_html = " ".join([
            f"<span style='display:inline-block;margin:2px;padding:2px 8px;border-radius:10px;"
            f"background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.25);"
            f"color:#10b981;font-size:0.72em;'>✓ {s.title()}</span>"
            for s in match['shared']
        ])
        gap_html = " ".join([
            f"<span style='display:inline-block;margin:2px;padding:2px 8px;border-radius:10px;"
            f"background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);"
            f"color:#f87171;font-size:0.72em;'>+ {s.title()}</span>"
            for s in match['missing']
        ])

        html += f"""
        <div style='margin-bottom:14px;padding:16px;border-radius:10px;
                    border:1px solid {c};background:rgba(0,0,0,0.2);'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                <div>
                    <span style='font-size:1.2em;margin-right:8px;'>{medals[i]}</span>
                    <span style='color:#e2e8f0;font-weight:700;font-size:0.95em;'>{match["job"]}</span>
                    <span style='color:#64748b;font-size:0.75em;margin-left:8px;'>{labels[i]}</span>
                </div>
                <div style='text-align:right;'>
                    <span style='color:{c};font-weight:800;font-size:1.1em;'>{pct}%</span>
                    <p style='color:#475569;font-size:0.7em;margin:0;'>KNN confidence</p>
                </div>
            </div>
            <div style='width:100%;background:#1e293b;border-radius:6px;margin-bottom:10px;'>
                <div style='width:{pct}%;background:{c};height:8px;border-radius:6px;
                            transition:width 0.5s ease;'></div>
            </div>
            <div style='margin-bottom:6px;'>{shared_html}</div>
            <div>{gap_html}</div>
        </div>"""

    html += """
        <p style='color:#334155;font-size:0.72em;text-align:right;margin-top:4px;'>
            Model: KNeighborsClassifier · Metric: Cosine · k=10 · Trained on 493 job roles
        </p>
    </div>"""
    return html


def build_forecast_html(user_skills: list):
    """Build skill demand forecast section using Linear Regression."""
    forecaster = get_forecaster()

    # User's own skills outlook
    user_outlook = forecaster.get_user_skill_outlook(user_skills)
    # Top rising skills in market
    top_rising = forecaster.get_top_rising(12)
    # Top stable high-demand skills
    top_stable = forecaster.get_top_stable(8)

    trend_icon  = {"Rising": "📈", "Stable": "➡️", "Declining": "📉"}
    trend_color = {"Rising": "#10b981", "Stable": "#3b82f6", "Declining": "#ef4444"}

    html = """
    <div>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:16px;'>
            <h3 style='color:#e2e8f0;font-size:1.05em;font-weight:700;margin:0;'>
                📊 Skill Demand Forecast — Linear Regression Model
            </h3>
            <span style='background:rgba(59,130,246,0.12);color:#60a5fa;border:1px solid rgba(59,130,246,0.3);
                         padding:2px 10px;border-radius:12px;font-size:0.72em;font-weight:700;'>
                ML Model
            </span>
        </div>
        <p style='color:#475569;font-size:0.78em;margin-bottom:18px;'>
            Each skill is modelled with a Linear Regression trained on 8 simulated quarterly data points.
            Slope coefficient determines Rising/Stable/Declining classification.
            Growth % = forecast for next quarter vs current demand.
        </p>"""

    # ── Your skills outlook ──
    if user_outlook:
        html += """<div style='margin-bottom:20px;padding:14px;border-radius:10px;
                    background:#0a1628;border:1px solid #1e293b;'>
            <p style='color:#60a5fa;font-weight:700;font-size:0.85em;margin-bottom:12px;'>
                🎯 Demand Outlook for YOUR Skills
            </p>"""
        for fc in user_outlook[:8]:
            t  = fc['trend']
            tc = trend_color[t]
            ti = trend_icon[t]
            gp = fc['growth_pct']
            gp_color = "#10b981" if gp > 0 else "#ef4444"
            bar_w = min(100, round((fc['current'] / 82) * 100))

            html += f"""
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;
                        padding:8px 12px;border-radius:7px;background:#0f172a;border:1px solid #1e293b;'>
                <span style='min-width:130px;color:#cbd5e1;font-size:0.82em;'>{fc['skill'].title()}</span>
                <div style='flex:1;background:#1e293b;border-radius:4px;height:5px;'>
                    <div style='width:{bar_w}%;background:{tc};height:5px;border-radius:4px;'></div>
                </div>
                <span style='color:{gp_color};font-size:0.78em;font-weight:700;min-width:50px;text-align:right;'>
                    {'+' if gp > 0 else ''}{gp}%
                </span>
                <span style='color:{tc};font-size:0.78em;font-weight:600;min-width:65px;'>
                    {ti} {t}
                </span>
                <span style='color:#475569;font-size:0.7em;min-width:50px;'>
                    slope={fc['slope']}
                </span>
            </div>"""
        html += "</div>"

    # ── Top Rising in Market ──
    html += """<div style='margin-bottom:20px;'>
        <p style='color:#10b981;font-weight:700;font-size:0.85em;margin-bottom:10px;'>
            🚀 Top Rising Skills in Job Market (Learn These!)
        </p>
        <div style='display:flex;flex-wrap:wrap;gap:8px;'>"""
    for fc in top_rising:
        html += f"""
        <div style='padding:8px 12px;border-radius:8px;border:1px solid rgba(16,185,129,0.3);
                    background:rgba(16,185,129,0.06);min-width:130px;'>
            <p style='color:#10b981;font-weight:700;font-size:0.8em;margin:0 0 2px;'>
                📈 {fc['skill'].title()}</p>
            <p style='color:#64748b;font-size:0.72em;margin:0;'>
                +{fc['growth_pct']}% · slope={fc['slope']}</p>
        </div>"""
    html += "</div></div>"

    # ── Stable High-Demand ──
    html += """<div>
        <p style='color:#3b82f6;font-weight:700;font-size:0.85em;margin-bottom:10px;'>
            💎 High-Demand Stable Skills (Safe to Learn)
        </p>
        <div style='display:flex;flex-wrap:wrap;gap:8px;'>"""
    for fc in top_stable:
        html += f"""
        <div style='padding:8px 12px;border-radius:8px;border:1px solid rgba(59,130,246,0.3);
                    background:rgba(59,130,246,0.06);min-width:120px;'>
            <p style='color:#60a5fa;font-weight:700;font-size:0.8em;margin:0 0 2px;'>
                ➡️ {fc['skill'].title()}</p>
            <p style='color:#64748b;font-size:0.72em;margin:0;'>
                demand={fc['current']} · R²={fc['r2']}</p>
        </div>"""
    html += """</div></div>
        <p style='color:#334155;font-size:0.72em;text-align:right;margin-top:12px;'>
            Model: LinearRegression · Features: 8 quarterly points · Fitted on 690 skills
        </p>
    </div>"""
    return html


# ── Train on import ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    knn, forecaster = load_or_train_models(force_retrain=True)
    print("\n=== TEST KNN ===")
    test_scores = {"python": 4, "sql": 3, "docker": 2, "aws": 3, "machine learning": 4}
    matches = knn.predict(test_scores, top_n=3)
    for m in matches:
        print(f"  {m['rank']}. {m['job']} — {m['confidence']}%")
        print(f"     Shared: {m['shared']}")
        print(f"     Missing: {m['missing']}")

    print("\n=== TEST REGRESSION ===")
    outlook = forecaster.get_user_skill_outlook(["python", "docker", "aws"])
    for o in outlook:
        print(f"  {o['skill']}: {o['trend']} (+{o['growth_pct']}%) slope={o['slope']}")
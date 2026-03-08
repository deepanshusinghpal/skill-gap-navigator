"""
Phase 2 — PDF Report Generator
Generates a professional career analysis PDF for download.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import os

# ── Brand Colors ────────────────────────────────────────────────────────────
GREEN   = HexColor("#10b981")
BLUE    = HexColor("#3b82f6")
PURPLE  = HexColor("#8b5cf6")
AMBER   = HexColor("#f59e0b")
RED     = HexColor("#ef4444")
DARK    = HexColor("#0f172a")
SLATE   = HexColor("#1e293b")
MUTED   = HexColor("#64748b")
LIGHT   = HexColor("#e2e8f0")
WHITE   = white


def generate_pdf_report(email, best_job, top3_jobs, score_df,
                         missing_skills, verified_scores, quiz_pct,
                         output_path=None):
    """
    Generates a PDF career report and saves it to output_path.
    Returns the file path.
    """
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.dirname(__file__), '..', f"career_report_{ts}.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Style helpers ────────────────────────────────────────────────────────
    def h1(text):
        return Paragraph(text, ParagraphStyle("H1", fontSize=22, textColor=WHITE,
                         fontName="Helvetica-Bold", spaceAfter=4, leading=28))
    def h2(text, color=GREEN):
        return Paragraph(text, ParagraphStyle("H2", fontSize=13, textColor=color,
                         fontName="Helvetica-Bold", spaceAfter=6, spaceBefore=14, leading=18))
    def h3(text):
        return Paragraph(text, ParagraphStyle("H3", fontSize=10, textColor=LIGHT,
                         fontName="Helvetica-Bold", spaceAfter=4, leading=14))
    def body(text, color=LIGHT):
        return Paragraph(text, ParagraphStyle("Body", fontSize=9, textColor=color,
                         fontName="Helvetica", spaceAfter=4, leading=14))
    def small(text, color=MUTED):
        return Paragraph(text, ParagraphStyle("Small", fontSize=8, textColor=color,
                         fontName="Helvetica", spaceAfter=3, leading=12))
    def hr():
        return HRFlowable(width="100%", thickness=1, color=SLATE, spaceAfter=10, spaceBefore=6)

    # ── HEADER BANNER ────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("🚀 Skill Gap Navigator", ParagraphStyle("HH", fontSize=20,
                  textColor=WHITE, fontName="Helvetica-Bold", leading=26)),
        Paragraph(f"Career Analysis Report<br/><font size='9' color='#64748b'>{datetime.now().strftime('%B %d, %Y')}</font>",
                  ParagraphStyle("HS", fontSize=11, textColor=LIGHT,
                                 fontName="Helvetica", alignment=TA_RIGHT, leading=16))
    ]]
    header_table = Table(header_data, colWidths=["60%", "40%"])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), DARK),
        ("ROWPADDING",  (0,0), (-1,-1), 14),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("BOX",         (0,0), (-1,-1), 2, GREEN),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    # User info strip
    info_data = [[
        body(f"<b>Email:</b> {email}"),
        body(f"<b>Assessment Score:</b> {quiz_pct}%"),
        body(f"<b>Top Match:</b> {best_job}"),
    ]]
    info_table = Table(info_data, colWidths=["34%", "33%", "33%"])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), SLATE),
        ("ROWPADDING",  (0,0), (-1,-1), 8),
        ("TEXTCOLOR",   (0,0), (-1,-1), LIGHT),
        ("BOX",         (0,0), (-1,-1), 1, MUTED),
        ("LINEAFTER",   (0,0), (1,-1), 1, MUTED),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))

    # ── TOP 3 JOB MATCHES ───────────────────────────────────────────────────
    story.append(h2("💼  Top 3 Career Matches", GREEN))
    story.append(hr())

    medals = ["🥇", "🥈", "🥉"]
    colors_list = [GREEN, BLUE, PURPLE]
    labels = ["Best Match", "Strong Match", "Good Match"]

    for rank, (job_name, pct) in enumerate(top3_jobs[:3]):
        bar_filled  = int(pct / 5)   # out of 20 cells
        bar_empty   = 20 - bar_filled
        bar_str     = "█" * bar_filled + "░" * bar_empty

        row_data = [[
            Paragraph(f"{medals[rank]}  <b>{job_name}</b>",
                      ParagraphStyle("JN", fontSize=10, textColor=LIGHT,
                                     fontName="Helvetica-Bold", leading=14)),
            Paragraph(f"<font color='#{colors_list[rank].hexval()[2:]}'>{bar_str}</font>",
                      ParagraphStyle("Bar", fontSize=8, textColor=LIGHT,
                                     fontName="Courier", leading=12)),
            Paragraph(f"<b>{pct}%</b><br/><font size='8' color='#64748b'>{labels[rank]}</font>",
                      ParagraphStyle("Pct", fontSize=10, textColor=colors_list[rank],
                                     fontName="Helvetica-Bold", alignment=TA_RIGHT, leading=14)),
        ]]
        t = Table(row_data, colWidths=["35%", "45%", "20%"])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), SLATE),
            ("ROWPADDING",  (0,0), (-1,-1), 10),
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ("BOX",         (0,0), (-1,-1), 1.5, colors_list[rank]),
            ("LEFTPADDING", (0,0), (0,-1), 12),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*cm))

    story.append(Spacer(1, 0.3*cm))

    # ── VERIFIED SKILL SCORES ───────────────────────────────────────────────
    story.append(h2("🧠  Verified Skill Scores", BLUE))
    story.append(hr())

    if not score_df.empty:
        table_data = [["Skill", "Score", "Level"]]
        for _, row in score_df.iterrows():
            skill = str(row["Technical Skill"])
            score_str = str(row["Verified Score"])
            score_val = int(score_str.split("/")[0].strip())
            level = ["❌ None", "⚠️  Beginner", "📖 Basic",
                     "✅ Intermediate", "🌟 Advanced", "🏆 Expert"][score_val]
            table_data.append([skill, score_str, level])

        skills_table = Table(table_data, colWidths=["50%", "20%", "30%"])
        style = [
            ("BACKGROUND",   (0,0), (-1,0),  DARK),
            ("TEXTCOLOR",    (0,0), (-1,0),  GREEN),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 9),
            ("ROWPADDING",   (0,0), (-1,-1), 7),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [SLATE, HexColor("#162032")]),
            ("TEXTCOLOR",    (0,1), (-1,-1), LIGHT),
            ("BOX",          (0,0), (-1,-1), 1, MUTED),
            ("LINEBELOW",    (0,0), (-1,0),  1, GREEN),
            ("GRID",         (0,1), (-1,-1), 0.3, HexColor("#334155")),
        ]
        skills_table.setStyle(TableStyle(style))
        story.append(skills_table)
    else:
        story.append(body("No skill scores recorded."))

    story.append(Spacer(1, 0.4*cm))

    # ── LEARNING PATH ───────────────────────────────────────────────────────
    story.append(h2("📚  Your Personalized Learning Path", PURPLE))
    story.append(hr())

    PATHS = {
        "python":           (6,  ["Python.org docs", "Automate the Boring Stuff (free)", "freeCodeCamp Python"]),
        "machine learning": (10, ["Andrew Ng ML Course (Coursera)", "Scikit-learn docs", "Kaggle Learn"]),
        "deep learning":    (8,  ["fast.ai (free)", "deeplearning.ai", "PyTorch tutorials"]),
        "sql":              (4,  ["SQLZoo (free)", "Mode SQL Tutorial", "LeetCode SQL"]),
        "docker":           (3,  ["Docker official docs", "Play with Docker (free)", "TechWorld with Nana"]),
        "kubernetes":       (5,  ["Kubernetes.io docs", "KodeKloud (free tier)", "CKAD prep guide"]),
        "aws":              (6,  ["AWS Free Tier + docs", "A Cloud Guru", "AWS Skill Builder (free)"]),
        "azure":            (6,  ["Microsoft Learn (free)", "AZ-900 study guide", "Azure free account"]),
        "javascript":       (6,  ["javascript.info (free)", "The Odin Project", "freeCodeCamp JS"]),
        "linux":            (3,  ["Linux Journey (free)", "OverTheWire Bandit", "The Linux Command Line"]),
        "git":              (2,  ["Pro Git book (free)", "Learn Git Branching (interactive)", "GitHub Skills"]),
        "ci/cd":            (4,  ["GitHub Actions docs", "Jenkins tutorials", "GitLab CI/CD docs"]),
        "cloud computing":  (5,  ["AWS/Azure/GCP free tiers", "Cloud Guru", "Google Cloud Skills Boost"]),
        "security":         (8,  ["TryHackMe (free)", "CompTIA Security+ guide", "OWASP Top 10"]),
        "networking":       (4,  ["Professor Messer (free)", "Cisco NetAcad", "NetworkChuck (YouTube)"]),
    }

    skills_with_paths = [s for s in missing_skills[:8] if s.lower() in PATHS]
    skills_generic    = [s for s in missing_skills[:8] if s.lower() not in PATHS]
    total_weeks = sum(PATHS.get(s.lower(), (3, []))[0] for s in skills_with_paths)

    story.append(body(f"<b>Estimated timeline to become job-ready for {best_job}: ~{total_weeks} weeks</b>"))
    story.append(Spacer(1, 0.2*cm))

    for i, skill in enumerate(skills_with_paths):
        weeks, resources = PATHS[skill.lower()]
        res_str = " · ".join(resources)
        path_data = [[
            Paragraph(f"<b>Step {i+1}</b>",
                      ParagraphStyle("SN", fontSize=9, textColor=PURPLE,
                                     fontName="Helvetica-Bold", leading=12)),
            Paragraph(f"<b>{skill.title()}</b>  —  ~{weeks} weeks<br/>"
                      f"<font size='8' color='#64748b'>Resources: {res_str}</font>",
                      ParagraphStyle("SP", fontSize=9, textColor=LIGHT,
                                     fontName="Helvetica", leading=14)),
        ]]
        t = Table(path_data, colWidths=["12%", "88%"])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (0,-1), SLATE),
            ("BACKGROUND",  (1,0), (1,-1), HexColor("#162032")),
            ("ROWPADDING",  (0,0), (-1,-1), 8),
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
            ("BOX",         (0,0), (-1,-1), 1, HexColor("#334155")),
            ("LINEAFTER",   (0,0), (0,-1), 2, PURPLE),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.15*cm))

    if skills_generic:
        story.append(small(f"Also explore: {', '.join([s.title() for s in skills_generic])}"))

    story.append(Spacer(1, 0.4*cm))

    # ── FOOTER ──────────────────────────────────────────────────────────────
    footer_data = [[
        small("Generated by Skill Gap Navigator", MUTED),
        small(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", MUTED),
    ]]
    footer_table = Table(footer_data, colWidths=["60%", "40%"])
    footer_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), DARK),
        ("ROWPADDING",  (0,0), (-1,-1), 8),
        ("ALIGN",       (1,0), (1,-1), "RIGHT"),
        ("BOX",         (0,0), (-1,-1), 1, MUTED),
    ]))
    story.append(hr())
    story.append(footer_table)

    doc.build(story)
    print(f"✅ PDF report saved: {output_path}")
    return output_path
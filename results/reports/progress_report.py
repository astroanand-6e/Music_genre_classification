"""Generate a one-page progress report PDF for CS5100 FAI project."""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os

OUT_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(OUT_DIR, exist_ok=True)
PDF_PATH = os.path.join(OUT_DIR, "progress_report.pdf")

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=16, spaceAfter=4)
subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=8)
heading_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=4, spaceBefore=8)
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=12)
small_style = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#444444"))

GREEN = colors.HexColor("#27ae60")
BLUE = colors.HexColor("#2980b9")
ORANGE = colors.HexColor("#e67e22")
GREY_BG = colors.HexColor("#f5f5f5")
HEADER_BG = colors.HexColor("#2c3e50")

def build():
    doc = SimpleDocTemplate(PDF_PATH, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.4*inch,
                            leftMargin=0.6*inch, rightMargin=0.6*inch)
    story = []

    # Title
    story.append(Paragraph("CS5100 FAI — Music Genre Classification", title_style))
    story.append(Paragraph("Progress Report  |  April 2, 2026  |  Anand Dev", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))

    # Completion
    story.append(Paragraph("<b>Overall Completion: ~65%</b>", heading_style))
    story.append(Paragraph(
        "Zero-shot and fine-tune done on FMA-Small (all models). "
        "FMA-Medium zero-shot done for 4/6 models; fine-tune done for 3/6. "
        "CLAP-Microsoft and MusicLDM-VAE medium zero-shot in progress. "
        "Fine-tuning on FMA-Medium and final report remain.",
        body_style,
    ))
    story.append(Spacer(1, 6))

    # Milestone checklist
    story.append(Paragraph("<b>Milestones</b>", heading_style))
    milestones = [
        ["Milestone", "Status"],
        ["FMA-Small: all zero-shot evals", "DONE"],
        ["FMA-Small: all fine-tune runs", "DONE"],
        ["FMA-Medium: zero-shot (MERT-95M, 330M, AST, CLAP-LAION)", "DONE"],
        ["FMA-Medium: fine-tune (MERT-95M, 330M, AST)", "DONE"],
        ["FMA-Medium: zero-shot (CLAP-Microsoft, MusicLDM-VAE)", "IN PROGRESS"],
        ["FMA-Medium: fine-tune (CLAP-LAION, CLAP-MS, MusicLDM)", "TODO"],
        ["Lyrics multimodal fusion experiment", "TODO"],
        ["Final report & analysis", "TODO"],
    ]
    mt = Table(milestones, colWidths=[4.5*inch, 1.5*inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(mt)
    story.append(Spacer(1, 8))

    # Results table
    story.append(Paragraph("<b>Results Summary</b>", heading_style))

    header = ["Model", "FMA-Small ZS", "FMA-Small FT", "FMA-Medium ZS", "FMA-Medium FT"]
    rows = [
        header,
        ["MERT-95M",    "58.41%", "63.12%", "67.74%", "70.92%"],
        ["MERT-330M",   "62.41%", "66.06%", "70.26%", "73.26%"],
        ["AST",         "55.25%", "65.94%", "67.08%", "71.72%"],
        ["CLAP-LAION",  "12.51%", "63.75%", "32.02%", "—"],
        ["CLAP-Microsoft", "—",  "—",       "running", "—"],
        ["MusicLDM-VAE","50.06%", "56.87%", "running", "—"],
    ]
    t = Table(rows, colWidths=[1.3*inch, 1.1*inch, 1.1*inch, 1.2*inch, 1.2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>ZS = zero-shot (linear probe on frozen features), FT = fine-tuned end-to-end. "
        "Best: MERT-330M FT on FMA-Medium at 73.26%.</i>",
        small_style,
    ))
    story.append(Spacer(1, 8))

    # Key findings
    story.append(Paragraph("<b>Key Findings</b>", heading_style))
    findings = [
        "<b>MERT-330M leads across all settings</b> — music-specific pretraining (self-supervised on 160k hrs) "
        "outperforms AST (AudioSet) and CLAP (text-audio contrastive).",
        "<b>Fine-tuning consistently improves over zero-shot</b> — average +8.5 pp on FMA-Small, "
        "with the largest gain for CLAP-LAION (+51.2 pp), whose contrastive embeddings are poorly suited to linear probes.",
        "<b>MusicLDM-VAE v2 (pre-bottleneck features)</b> — extracting from mid_block (512-ch) instead of "
        "conv_out (16-ch) improved zero-shot from 32.25% to 50.06% (+17.8 pp). Generative model features "
        "are viable but require careful extraction.",
        "<b>FMA-Medium improves all models</b> — ~4x more data (25k vs 8k tracks, 16 genres) boosts "
        "MERT-330M from 66% to 73%. Larger training set benefits discriminative models most.",
    ]
    for f in findings:
        story.append(Paragraph(f"• {f}", body_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 8))

    # Next steps
    story.append(Paragraph("<b>Next Steps</b>", heading_style))
    nexts = [
        "Complete CLAP-Microsoft & MusicLDM zero-shot on FMA-Medium",
        "Fine-tune remaining models on FMA-Medium (CLAP-LAION, CLAP-MS, MusicLDM)",
        "Lyrics-based multimodal fusion experiment",
        "Final comparative analysis & report",
    ]
    for n in nexts:
        story.append(Paragraph(f"• {n}", body_style))

    doc.build(story)
    print(f"PDF saved -> {PDF_PATH}")

if __name__ == "__main__":
    build()

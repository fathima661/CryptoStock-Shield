from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tempfile
import os


# ==========================================
# 🎨 STYLES
# ==========================================
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "title",
    parent=styles["Heading1"],
    textColor=colors.white,
    fontSize=18,
    spaceAfter=10
)

section_style = ParagraphStyle(
    "section",
    parent=styles["Heading2"],
    textColor=colors.cyan,
    fontSize=14,
    spaceAfter=8
)

normal_style = ParagraphStyle(
    "normal",
    parent=styles["Normal"],
    textColor=colors.white,
    fontSize=10
)


# ==========================================
# 🧱 BACKGROUND
# ==========================================
def add_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColorRGB(0.07, 0.09, 0.15)
    canvas.rect(0, 0, A4[0], A4[1], fill=1)
    canvas.restoreState()


# ==========================================
# 📊 CHART GENERATOR
# ==========================================
def generate_chart(risk, anomaly, confidence):
    fig = plt.figure()

    labels = ["Risk", "Anomaly", "Confidence"]
    values = [
        risk * 100,
        anomaly * 100,
        confidence * 100
    ]

    plt.bar(labels, values)
    plt.title("Model Metrics")
    plt.ylabel("Score (%)")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp.name, bbox_inches="tight")
    plt.close(fig)

    return temp.name


# ==========================================
# 📄 MAIN PDF BUILDER
# ==========================================
def generate_pdf(prediction):

    scan = prediction.scan
    asset = scan.asset

    file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(file_path, pagesize=A4)

    elements = []

    # ======================================
    # HEADER
    # ======================================
    elements.append(Paragraph("CryptoStock Shield AI Report", title_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Asset: {getattr(asset, 'symbol', 'N/A')}", normal_style))
    elements.append(Paragraph(f"Market: {getattr(asset, 'market_type', 'N/A')}", normal_style))
    elements.append(Paragraph(f"Generated: {scan.created_at}", normal_style))

    elements.append(Spacer(1, 20))

    # ======================================
    # RISK SUMMARY
    # ======================================
    elements.append(Paragraph("Risk Summary", section_style))

    risk_percent = round((prediction.risk_score or 0) * 100, 2)
    anomaly_percent = round((prediction.anomaly_score or 0) * 100, 2)
    confidence_percent = round((prediction.confidence or 0) * 100, 2)

    summary_data = [
        ["Metric", "Value"],
        ["Risk Score", f"{risk_percent}%"],
        ["Confidence", f"{confidence_percent}%"],
        ["Anomaly Score", f"{anomaly_percent}%"],
        ["Prediction", "Manipulated" if prediction.is_manipulated else "Normal"],
        ["Time Window", prediction.predicted_time_window or "N/A"],
    ]

    table = Table(summary_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    elements.append(table)

    # 🔥 Risk Interpretation
    elements.append(Spacer(1, 10))

    if prediction.risk_score >= 0.7:
        risk_text = "⚠️ High probability of coordinated market manipulation detected."
    elif prediction.risk_score >= 0.4:
        risk_text = "Moderate irregular activity detected. Monitor closely."
    else:
        risk_text = "Market behavior appears normal."

    elements.append(Paragraph(risk_text, normal_style))
    elements.append(Spacer(1, 20))

    # ======================================
    # CHART (SAFE)
    # ======================================
    elements.append(Paragraph("Model Metrics Chart", section_style))

    chart_path = None

    try:
        chart_path = generate_chart(
            prediction.risk_score or 0,
            prediction.anomaly_score or 0,
            prediction.confidence or 0
        )

        elements.append(Image(chart_path, width=5 * inch, height=3 * inch))

    except Exception:
        elements.append(Paragraph("Chart unavailable", normal_style))

    elements.append(PageBreak())

    # ======================================
    # AI EXPLANATION
    # ======================================
    elements.append(Paragraph("AI Explanation", section_style))

    explanation = getattr(prediction, "explanation", None)

    if explanation and getattr(explanation, "summary", None):
        elements.append(Paragraph(explanation.summary, normal_style))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Top Factors:", normal_style))

        features = explanation.feature_importance or {}
        for key, value in features.items():
            elements.append(Paragraph(f"- {key}: {value}", normal_style))
    else:
        elements.append(Paragraph("No explanation available", normal_style))

    elements.append(PageBreak())

    # ======================================
    # INPUT SNAPSHOT
    # ======================================
    elements.append(Paragraph("Input Data Snapshot", section_style))

    snapshot = getattr(prediction, "input_snapshot", {}) or {}

    snapshot_data = [["Feature", "Value"]]

    for k, v in snapshot.items():
        try:
            val = round(float(v), 4)
        except Exception:
            val = str(v)

        snapshot_data.append([
            Paragraph(str(k), normal_style),
            Paragraph(str(val)[:100], normal_style)
        ])

    table2 = Table(snapshot_data)
    table2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ]))

    elements.append(table2)

    # ======================================
    # BUILD PDF
    # ======================================
    doc.build(elements, onFirstPage=add_bg, onLaterPages=add_bg)

    # ======================================
    # CLEANUP
    # ======================================
    if chart_path and os.path.exists(chart_path):
        os.remove(chart_path)

    return file_path
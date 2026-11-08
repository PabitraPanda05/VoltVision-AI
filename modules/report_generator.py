"""PDF report generation for PowerLine_AI using ReportLab."""
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def generate_inspection_pdf(record: dict) -> bytes:
    """Generate a PDF report from one inspection-history record."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title=f"PowerLine_AI Inspection Report #{record['id']}",
        author="PowerLine_AI",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PowerLineTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=19, leading=23,
        textColor=colors.HexColor("#123456"), alignment=TA_CENTER,
        spaceAfter=5 * mm,
    ))

    story = [
        Paragraph("PowerLine_AI", styles["PowerLineTitle"]),
        Paragraph("Power-Line Component Inspection Report", styles["Heading2"]),
        Spacer(1, 4 * mm),
    ]

    details = [
        ["Inspection ID", str(record["id"])],
        ["Date and time", str(record["created_at"])],
        ["Inspection type", str(record["inspection_type"])],
        ["Source file", str(record["source_filename"])],
        ["Status", str(record.get("status", "Completed"))],
        ["Confidence threshold", f"{float(record['confidence_threshold']):.2f}"],
        ["Processing time", f"{float(record['processing_seconds']):.2f} seconds"],
    ]
    if record.get("inspection_type") == "Video":
        details += [
            ["Total frames", str(record.get("total_frames") or 0)],
            ["Frames analyzed", str(record.get("analyzed_frames") or 0)],
            ["Frame interval", str(record.get("frame_interval") or "—")],
        ]

    table = Table(details, colWidths=[48 * mm, 118 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D3DD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [table, Spacer(1, 7 * mm),
              Paragraph("Detection Summary", styles["Heading2"]),
              Paragraph(
                  f"Total component detections: <b>{int(record['detection_count'])}</b>",
                  styles["BodyText"],
              ),
              Spacer(1, 3 * mm)]

    counts = record.get("component_counts") or {}
    if counts:
        rows = [["Component", "Detection count"]]
        rows += [[str(name), str(int(count))] for name, count in counts.items()]
        counts_table = Table(rows, colWidths=[115 * mm, 51 * mm], repeatRows=1)
        counts_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1269DC")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D3DD")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F4F7FA")]),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(counts_table)
    else:
        story.append(Paragraph(
            "No components were detected above the selected confidence threshold.",
            styles["BodyText"],
        ))

    detections = record.get("detections") or []
    if detections:
        story += [Spacer(1, 7 * mm),
                  Paragraph("Detection Details", styles["Heading2"])]
        rows = [["#", "Component", "Confidence"]]
        for i, item in enumerate(detections[:100], start=1):
            rows.append([
                str(i), str(item.get("component", "Unknown")),
                f"{float(item.get('confidence', 0)):.3f}",
            ])
        detail_table = Table(rows, colWidths=[12 * mm, 112 * mm, 42 * mm],
                             repeatRows=1)
        detail_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1269DC")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C8D3DD")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(detail_table)

    story += [
        Spacer(1, 8 * mm),
        Paragraph(
            "Scope note: This report summarizes AI component detections only. "
            "It does not certify electrical safety, identify electrical defects, "
            "or replace inspection by a qualified professional.",
            styles["Italic"],
        ),
    ]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(18 * mm, 10 * mm, "PowerLine_AI • Inspection Record")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()

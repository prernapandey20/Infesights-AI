from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def create_executive_pdf(report: str) -> bytes:
	buffer = BytesIO()
	document = SimpleDocTemplate(
		buffer,
		pagesize=letter,
		leftMargin=0.75 * inch,
		rightMargin=0.75 * inch,
		topMargin=0.7 * inch,
		bottomMargin=0.7 * inch,
		title="Executive Anomaly Post-Mortem",
	)
	styles = getSampleStyleSheet()
	styles.add(
		ParagraphStyle(
			name="ReportHeading",
			parent=styles["Heading2"],
			textColor=colors.HexColor("#17324D"),
			spaceBefore=12,
			spaceAfter=6,
		)
	)
	styles.add(
		ParagraphStyle(
			name="ReportBody",
			parent=styles["BodyText"],
			fontName="Helvetica",
			fontSize=9.5,
			leading=14,
			alignment=TA_LEFT,
			spaceAfter=6,
		)
	)
	story = [
		Paragraph("Executive Anomaly Post-Mortem", styles["Title"]),
		Spacer(1, 12),
	]
	for line in report.splitlines():
		plain_text = line.replace("**", "").replace("`", "")
		plain_text = plain_text.encode("ascii", "replace").decode("ascii")
		if not plain_text.strip():
			story.append(Spacer(1, 5))
			continue
		if plain_text.startswith("### "):
			story.append(Paragraph(escape(plain_text[4:]), styles["ReportHeading"]))
		elif plain_text.startswith("#### "):
			story.append(Paragraph(escape(plain_text[5:]), styles["Heading3"]))
		else:
			story.append(Paragraph(escape(plain_text), styles["ReportBody"]))
	document.build(story)
	return buffer.getvalue()
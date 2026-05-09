from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os

def draw_wrapped_text(c, text, x, y, max_width):
    words = text.split(' ')
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if c.stringWidth(' '.join(current_line), "Helvetica", 11) > max_width:
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    for line in lines:
        c.drawString(x, y, line)
        y -= 15
    return y

def generate_pdf(data, file_path, patient_name="N/A", patient_id="N/A"):
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(colors.HexColor("#38bdf8"))
    c.drawString(50, 800, "MEDICAL AI DIAGNOSTIC REPORT")
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    c.drawString(50, 770, f"Patient Name: {patient_name}")
    c.drawString(50, 750, f"Patient ID: {patient_id}")
    c.drawString(400, 750, f"Date: {os.popen('date /t').read().strip()}")
    
    c.setStrokeColor(colors.grey)
    c.line(50, 735, 550, 735)

    # Diagnostic Results
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 710, "DIAGNOSTIC RESULTS")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, 685, f"Scan Modality: {data.get('type', 'N/A')}")
    c.drawString(50, 665, f"AI Prediction: {data.get('prediction', 'N/A')}")
    c.drawString(50, 645, f"Confidence Score: {data.get('confidence', '0')}%")

    y = 615
    
    # Disease Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Disease Summary & Details:")
    y -= 20
    c.setFont("Helvetica", 11)
    explanation = data.get('disease_explanation', 'No additional summary available.')
    y = draw_wrapped_text(c, explanation, 50, y, 500)
    
    y -= 10
    c.line(50, y, 550, y)
    y -= 20

    # Image Evidence
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "VISUAL EVIDENCE")
    y -= 150
    
    # Drawing Images if they exist
    orig_path = data.get("original_img_path")
    box_path = data.get("boxed_full_path")
    
    img_width = 180
    img_height = 140
    
    if orig_path and os.path.exists(orig_path):
        c.drawImage(orig_path, 50, y, width=img_width, height=img_height, preserveAspectRatio=True)
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(50, y - 15, "Original Scan")
    
    if box_path and os.path.exists(box_path):
        c.drawImage(box_path, 300, y, width=img_width, height=img_height, preserveAspectRatio=True)
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(300, y - 15, "AI Detected Region")

    y -= 45 # Space for next section

    # Treatment Protocol
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "TREATMENT & MEDICATION PROTOCOL")
    y -= 20
    
    c.setFont("Helvetica", 11)
    for med in data.get('medications', []):
        c.drawString(60, y, f"• {med['medicine']} ({med['dosage']}) - {med['timing']}")
        y -= 20
        if y < 100:
            c.showPage()
            y = 750

    # Disclaimer
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.grey)
    c.drawString(50, y-40, "Disclaimer: Consult a certified doctor before taking any medication.")
    
    c.save()

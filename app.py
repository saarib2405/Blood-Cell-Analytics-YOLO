import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- 1. SETUP & CONFIG ---
st.set_page_config(
    page_title="AI Blood Cell Analytics",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Medical Theme (following Phase 2 guidelines)
st.markdown("""
<style>
    /* Add subtle background and medical-themed accents */
    .stApp {
        background-color: #0b1117;
    }
    .main-header {
        text-align: center;
        color: #e53e3e; /* Red accent for blood theme */
        font-weight: 700;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1a202c;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid #e53e3e;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #a0aec0;
        font-style: italic;
        margin-top: 50px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load YOLOv11 model with caching to prevent reloading on user interactions."""
    model_path = Path("runs/detect/blood_cell_model/weights/best.pt")
    if not model_path.exists():
        st.warning(f"Model not found at {model_path}. Please train the model first.")
        # Fallback to base model for development without crashing, if desired
        return None
    return YOLO(model_path)

model = load_model()

# --- 3. HELPER FUNCTIONS ---
def generate_pdf_report(original_img, annotated_img, counts_dict, timestamp, barchart_buf, piechart_buf):
    """Generates a 2-page PDF report using reportlab"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # --- Page 1 ---
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2.0, height - 50, "Blood Cell Analysis Report")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Generated on: {timestamp}")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 120, "1. Processed Image Summary")
    
    # Save annotated image to a temporary buffer for ReportLab
    annot_img_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    annot_img_buf = io.BytesIO()
    annot_img_pil.save(annot_img_buf, format='PNG')
    annot_img_buf.seek(0)
    
    c.drawImage(ImageReader(annot_img_buf), 50, height - 420, width=500, height=281, preserveAspectRatio=True)
    
    # Table of counts
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 460, "2. Cell Count Summary")
    
    c.setFont("Helvetica", 12)
    y_pos = height - 490
    for cell_type, count in counts_dict.items():
        c.drawString(70, y_pos, f"- {cell_type}: {count}")
        y_pos -= 25

    c.showPage()
    
    # --- Page 2 ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "3. Distribution Charts")
    
    # Draw Bar Chart
    c.drawImage(ImageReader(barchart_buf), 50, height - 350, width=400, height=250, preserveAspectRatio=True)
    
    # Draw Pie Chart
    c.drawImage(ImageReader(piechart_buf), 50, height - 650, width=400, height=250, preserveAspectRatio=True)
    
    # Disclaimer
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawCentredString(width / 2.0, 30, "DISCLAIMER: This is an AI-assisted tool. Not for clinical diagnosis.")
    
    c.save()
    buffer.seek(0)
    return buffer


def create_charts(counts_dict):
    """Generates matplotlib bar and pie charts and returns their buffers."""
    labels = list(counts_dict.keys())
    values = list(counts_dict.values())
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    # Bar Chart
    fig_bar, ax_bar = plt.subplots(figsize=(6,4), facecolor='#0b1117')
    ax_bar.bar(labels, values, color=colors)
    ax_bar.set_facecolor('#0b1117')
    ax_bar.tick_params(colors='white')
    for spine in ax_bar.spines.values():
        spine.set_edgecolor('white')
    ax_bar.set_title("Cell Count Distribution", color='white')
    plt.tight_layout()
    bar_buf = io.BytesIO()
    fig_bar.savefig(bar_buf, format='png', facecolor='#0b1117')
    bar_buf.seek(0)
    plt.close(fig_bar)
    
    # Pie Chart
    fig_pie, ax_pie = plt.subplots(figsize=(6,4), facecolor='#0b1117')
    if sum(values) > 0:
        ax_pie.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color':"w"})
        ax_pie.axis('equal')
    ax_pie.set_title("Cell Percentage", color='white')
    pie_buf = io.BytesIO()
    fig_pie.savefig(pie_buf, format='png', facecolor='#0b1117')
    pie_buf.seek(0)
    plt.close(fig_pie)
    
    return bar_buf, pie_buf


# --- 4. APP LAYOUT & LOGIC ---

# Header
st.markdown("<h1 class='main-header'>🩸 AI-Powered Blood Cell Analytics</h1>", unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.header("⚙️ Configuration & Upload")
    uploaded_file = st.file_uploader("Upload Blood Smear Image", type=["jpg", "png", "jpeg"])
    
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    
    st.markdown("<div class='disclaimer'>DISCLAIMER: This is an AI-assisted tool meant for research and demonstration purposes. It is NOT intended for clinical diagnosis.</div>", unsafe_allow_html=True)


if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    # Convert to OpenCV format (BGR)
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
        if model is None:
            st.error("Model is not loaded. Cannot perform inference.")
        else:
            with st.spinner('Running AI Analysis...'):
                # Inference
                results = model.predict(img_cv, imgsz=640, conf=conf_threshold)
                result = results[0]
                
                # Get annotated image
                annotated_img = result.plot()
                
                # Setup Counts
                # Class indices: 0: Platelets, 1: RBC, 2: WBC (from data.yaml)
                counts = {"Platelets": 0, "RBC": 0, "WBC": 0}
                names = result.names
                for c in result.boxes.cls:
                    class_name = names[int(c)]
                    if class_name in counts:
                        counts[class_name] += 1
                
                st.session_state['counts'] = counts
                st.session_state['annotated_img'] = annotated_img
                st.session_state['original_img'] = img_cv

# Dashboard Rendering if analysis is done
if 'counts' in st.session_state:
    st.markdown("---")
    st.subheader("🔬 Analysis Results")
    
    counts = st.session_state['counts']
    
    # 1. Metric Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>🔴 RBC Count</h3><h2>{counts.get('RBC', 0)}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>⚪ WBC Count</h3><h2>{counts.get('WBC', 0)}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>🟣 Platelet Count</h3><h2>{counts.get('Platelets', 0)}</h2></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Images Side by Side
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        # Convert original img_cv BGR back to RGB for streamlit
        orig_disp = cv2.cvtColor(st.session_state['original_img'], cv2.COLOR_BGR2RGB)
        st.image(orig_disp, caption="Original Image", use_container_width=True)
    with img_col2:
        annot_disp = cv2.cvtColor(st.session_state['annotated_img'], cv2.COLOR_BGR2RGB)
        st.image(annot_disp, caption="Annotated Image", use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Charts
    bar_buf, pie_buf = create_charts(counts)
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.image(bar_buf, use_container_width=True)
    with chart_col2:
        st.image(pie_buf, use_container_width=True)
        
    # 4. Report Generation
    st.markdown("---")
    st.subheader("📄 Export Results")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf_buffer = generate_pdf_report(
        st.session_state['original_img'], 
        st.session_state['annotated_img'], 
        counts, 
        timestamp,
        bar_buf, 
        pie_buf
    )
    
    st.download_button(
        label="📥 Download Lab Report (PDF)",
        data=pdf_buffer,
        file_name=f"blood_report_{timestamp.replace(':', '-')}.pdf",
        mime="application/pdf",
        type="primary"
    )
else:
    # Initial state screen before upload/analysis
    st.info("👈 Please upload an image from the sidebar and click 'Analyze Image' to see results here.")

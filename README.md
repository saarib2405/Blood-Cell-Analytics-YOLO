# AI-Powered Blood Cell Analytics 🩸

This repository contains a full-stack Streamlit application integrated with YOLOv11 for robust Blood Cell Detection and Counts (Platelets, RBC, WBC). The application allows for direct inference via uploaded smears and comprehensive automated reporting.

## Project Structure
- `app.py`: Streamlit main dashboard implementation
- `train.py`: Local YOLOv11 training script utilizing CUDA optimization
- `requirements.txt`: Locked dependencies
- `data.yaml`: Dataset mapping
- `runs/detect/blood_cell_model/weights/best.pt`: Model weights (generated post-training)

## Running Locally

1. Create a Python Virtual Environment:
```bash
python -m venv venv
```
2. Activate and install:
```bash
# Windows
.\venv\Scripts\activate
# Unix
source venv/bin/activate

pip install -r requirements.txt
```
3. Run the App:
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment
This repository is configured to run effortlessly out-of-the-box on Streamlit Cloud. 
1. Push this code to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Deploy directly targeting `app.py` as the execution entrypoint. The `.streamlit/config.toml` inherently applies the optimized medical Dark Theme.

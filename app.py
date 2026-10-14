from pathlib import Path
import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

st.set_page_config(page_title="Oral Cancer Triage AI", page_icon="🩺", layout="centered")

st.title("🩺 Oral Cancer Safety-First Triage AI")
st.caption("Clinical Triage Portal • Calibrated for High Sensitivity & Low False Alarms")

# 1. Dynamically find ANY .pt file in your muko-ai3 GitHub repo folder
BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_NAMES = [
    "best_oral_triage_b0_calibrated.pt",
    "best_oral_triage_b0_calibrated .pt",
    "best_oral_triage_b0_balanced.pt",
    "best_oral_triage_b0_balanced .pt",
    "best_oral_triage_b0_antishortcut.pt",
]

MODEL_PATH = None
for name in CANDIDATE_NAMES:
    candidate = BASE_DIR / name
    if candidate.exists():
        MODEL_PATH = candidate
        break

# Fallback: automatically grab any .pt file anywhere in the repo
if MODEL_PATH is None:
    pt_files = list(BASE_DIR.rglob("*.pt"))
    if pt_files:
        MODEL_PATH = pt_files[0]

# Safety-First Calibrated Thresholds (< 35% Green | 35%-65% Amber | >= 65% Red)
GREEN_MAX = 0.35
AMBER_MAX = 0.65

@st.cache_resource
def load_triage_model():
    if MODEL_PATH is None or not MODEL_PATH.exists():
        files_found = [p.name for p in BASE_DIR.iterdir()]
        st.error(
            f"❌ No `.pt` model file found in `{BASE_DIR}`!\n\n"
            f"**Files currently in your `muko-ai3` GitHub repo:** `{files_found}`\n\n"
            f"👉 Please upload `best_oral_triage_b0_calibrated.pt` to your `muko-ai3` repo on GitHub."
        )
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    state_dict = torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

model, device = load_triage_model()

eval_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tab1, tab2 = st.tabs(["📁 Upload Mouth Photo", "📷 Take Live Phone Photo"])
uploaded_file = tab1.file_uploader("Select a JPG or PNG image", type=["jpg", "jpeg", "png"])
camera_file = tab2.camera_input("Take a well-lit photo of the inner mouth")

image_source = uploaded_file if uploaded_file is not None else camera_file

if image_source is not None:
    raw_img = Image.open(image_source).convert("RGB")
    
    # Strip 8% outer border to avoid retractor, glove, and border artifacts
    w, h = raw_img.size
    cropped_img = raw_img.crop((int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92)))

    with st.spinner("Analyzing mucosal tissue & generating Grad-CAM..."):
        t_full = eval_transform(cropped_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            t_flip = torch.flip(t_full, dims=[3])
            
            # Apply Temperature Scaling (T = 2.0) + Average of original and flipped views
            p1 = torch.sigmoid(model(t_full) / 2.0)
            p2 = torch.sigmoid(model(t_flip) / 2.0)
            prob = float(((p1 + p2) / 2.0).cpu().item())

        disp_pil = cropped_img.resize((260, 260))
        rgb_float = np.float32(disp_pil) / 255.0
        
        targets_for_cam = [BinaryClassifierOutputTarget(1)]
        with GradCAM(model=model, target_layers=[model.conv_head]) as cam:
            grayscale_cam = cam(input_tensor=t_full, targets=targets_for_cam)[0, :]
        cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # 3-Tier Clinical Traffic Light
    if prob < GREEN_MAX:
        st.success(
            f"### 🟢 GREEN: Low Risk / Normal Tissue\n"
            f"**Risk Score:** `{prob*100:.1f}%` (Threshold < 35%)\n\n"
            f"**Action:** No high-risk visual mucosal abnormalities detected. Maintain routine oral hygiene."
        )
    elif prob < AMBER_MAX:
        st.warning(
            f"### 🟡 AMBER: Mucosal Variation / Suspicious Ulcer\n"
            f"**Risk Score:** `{prob*100:.1f}%` (Threshold 35% – 65%)\n\n"
            f"**Action:** Atypical tissue or lighting glare detected. If an ulcer, sore, or patch persists for more than 14 days, a clinical dental examination is advised."
        )
    else:
        st.error(
            f"### 🔴 RED: High-Risk Lesion Detected\n"
            f"**Risk Score:** `{prob*100:.1f}%` (Threshold ≥ 65%)\n\n"
            f"**Action:** High visual suspicion of malignancy. Immediate specialist biopsy and clinical referral recommended."
        )

    st.info(
        "**Clinical Disclaimer:** This software is a triage screening demonstration tool. "
        "It does not replace histopathological biopsy or professional clinical diagnosis."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(disp_pil, caption="Analyzed Oral Region", use_container_width=True)
    with col2:
        st.image(cam_overlay, caption="Grad-CAM Lesion Heatmap", use_container_width=True)

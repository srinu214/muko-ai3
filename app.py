import os
import glob
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

BASE_DRIVE = '/content/drive/MyDrive/oral_ai'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the newly calibrated model (falls back to balanced if calibrated isn't saved yet)
cal_path = f"{BASE_DRIVE}/models/best_oral_triage_b0_calibrated.pt"
bal_path = f"{BASE_DRIVE}/models/best_oral_triage_b0_balanced.pt"
active_model_path = cal_path if os.path.exists(cal_path) else bal_path
print(f"Testing with model: {os.path.basename(active_model_path)}")

model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=1).to(device)
model.load_state_dict(torch.load(active_model_path, map_location=device, weights_only=True))
model.eval()

eval_tf = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Locate Tp1, Tp2, Tpp2, Tpp3, Tp4 across Colab directories
target_prefixes = ['Tp1', 'Tp2', 'Tpp2', 'Tpp3', 'Tp4']
found_Paths = []

search_dirs = ['/content', '/content/clean/custom_healthy', BASE_DRIVE]
for prefix in target_prefixes:
    matched = []
    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                # Match exact stem (case-insensitive) regardless of extension (.jpg, .jpeg, .png)
                stem, ext = os.path.splitext(f)
                if stem.lower() == prefix.lower() and ext.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    matched.append(os.path.join(d, f))
    if matched:
        found_Paths.append((prefix, matched[0]))
    else:
        print(f"⚠️ Could not find '{prefix}' in {search_dirs}. Check exact spelling/folder.")

# 3. Evaluate & Plot all found images
if not found_Paths:
    print("No matching files found! Run `!ls /content` to check exact filenames.")
else:
    GREEN_MAX = 0.30
    AMBER_MAX = 0.65

    n = len(found_Paths)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7))
    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    targets_for_cam = [BinaryClassifierOutputTarget(1)]

    for idx, (label_name, img_path) in enumerate(found_Paths):
        raw_img = Image.open(img_path).convert('RGB')
        w, h = raw_img.size
        cropped_img = raw_img.crop((int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92)))

        t_full = eval_tf(cropped_img).unsqueeze(0).to(device)
        t_flip = torch.flip(t_full, dims=[3])

        with torch.no_grad():
            # Apply Temperature Scaling (T = 2.0) + Average of original and flipped
            p1 = torch.sigmoid(model(t_full) / 2.0)
            p2 = torch.sigmoid(model(t_flip) / 2.0)
            prob = float(((p1 + p2) / 2.0).cpu().item())

        # Assign Traffic Light
        if prob < GREEN_MAX:
            status, color = "GREEN (Safe)", "green"
        elif prob < AMBER_MAX:
            status, color = "AMBER (Checkup)", "darkorange"
        else:
            status, color = "RED (Urgent)", "red"

        # Compute Grad-CAM
        disp_pil = cropped_img.resize((260, 260))
        rgb_float = np.float32(disp_pil) / 255.0
        with GradCAM(model=model, target_layers=[model.conv_head]) as cam:
            grayscale_cam = cam(input_tensor=t_full, targets=targets_for_cam)[0, :]
        cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        # Top row: Cropped Input Image
        axes[0, idx].imshow(disp_pil)
        axes[0, idx].set_title(f"{label_name}\n{status}\nScore: {prob*100:.1f}%", fontsize=11, fontweight='bold', color=color)
        axes[0, idx].axis('off')

        # Bottom row: Grad-CAM Heatmap
        axes[1, idx].imshow(cam_overlay)
        axes[1, idx].set_title("Grad-CAM Focus", fontsize=10)
        axes[1, idx].axis('off')

        print(f"Image: {label_name:5s} ({os.path.basename(img_path)}) -> Score: {prob*100:5.1f}% -> {status}")

    plt.tight_layout()
    plt.show()
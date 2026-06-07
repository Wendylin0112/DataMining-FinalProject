import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------
# 載入 metadata
# -----------------------------
metadata = joblib.load("models/metadata.joblib")

components = metadata["components"]
component_to_idx = metadata["component_to_idx"]
idx_to_component = {v: k for k, v in component_to_idx.items()}

image_size = metadata["image_size"]
arch = metadata["arch"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 建立 model
# -----------------------------
def create_model(arch, num_classes):

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)

        in_features = model.classifier[-1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_features, num_classes)
        )

    elif arch == "resnet18":
        model = models.resnet18(weights=None)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


# -----------------------------
# 載入 component model
# -----------------------------
component_model = create_model(
    arch,
    len(components)
)

component_model.load_state_dict(
    torch.load(
        "models/component_model.pt",
        map_location=device
    )
)

component_model.to(device)
component_model.eval()

# -----------------------------
# 載入 defect model
# -----------------------------
defect_model = create_model(
    arch,
    1
)

defect_model.load_state_dict(
    torch.load(
        "models/defect_global.pt",
        map_location=device
    )
)

defect_model.to(device)
defect_model.eval()

# -----------------------------
# transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -----------------------------
# UI
# -----------------------------
st.title("UAV 電力設備異常辨識")

uploaded = st.file_uploader(
    "上傳圖片",
    type=["jpg", "jpeg", "png"]
)

threshold = st.slider(
    "判定門檻",
    0.0,
    1.0,
    0.5
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.image(image, width=300)

    x = transform(image).unsqueeze(0).to(device)

    # -----------------------------
    # component prediction
    # -----------------------------
    with torch.no_grad():

        component_logits = component_model(x)

        component_pred = component_logits.argmax(1).item()

        component_name = idx_to_component[component_pred]

    # -----------------------------
    # defect prediction
    # -----------------------------
    with torch.no_grad():

        defect_logits = defect_model(x)

        defect_score = torch.sigmoid(
            defect_logits
        ).item()

    status = "BAD / 損壞" if defect_score >= threshold else "GOOD / 正常"

    # -----------------------------
    # 顯示結果
    # -----------------------------
    st.subheader("預測結果")

    st.write(f"元件類型：{component_name}")

    st.write(f"異常分數：{defect_score:.4f}")

    if defect_score >= threshold:
        st.error(status)
    else:
        st.success(status)
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_MODEL_DIR = "hybrid_deep_models_p097"

COMPONENT_DISPLAY_NAMES = {
    "vari-grip": "vari-grip 可調式夾線條",
    "lightning-rod-suspension": "lightning-rod-suspension 避雷線懸吊裝置",
    "polymer-insulator-upper-shackle": "polymer-insulator-upper-shackle 絕緣子上部鉤環",
    "glass-insulator": "glass-insulator 玻璃絕緣子",
    "yoke-suspension": "yoke-suspension 聯軛懸吊組件",
}

# Thresholds used to create the final best submission:
# test_submission_hybrid_deep_p097_comp_balanced.csv
FINAL_COMPONENT_THRESHOLDS = {
    "lightning-rod-suspension": 0.40,
    "polymer-insulator-upper-shackle": 0.50,
    "yoke-suspension": 0.55,
}


def create_model(arch, num_classes, dropout=0.25):
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {arch}")
    return model


def make_eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_state_dict(model, path, device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading final hybrid model...")
def load_hybrid_model(model_dir):
    model_dir = Path(model_dir)
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    components = metadata["components"]
    component_to_idx = metadata["component_to_idx"]
    idx_to_component = {index: component for component, index in component_to_idx.items()}
    arch = metadata["arch"]
    image_size = metadata["image_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    component_model = create_model(arch, len(components))
    component_model = load_state_dict(component_model, model_dir / "component_model.pt", device)

    defect_models = {}
    routes = ["global", *metadata.get("specialist_components", [])]
    for route in routes:
        model_path = model_dir / f"defect_{route}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing defect model: {model_path}")
        defect_model = create_model(arch, 1)
        defect_models[route] = load_state_dict(defect_model, model_path, device)

    return {
        "metadata": metadata,
        "device": device,
        "component_model": component_model,
        "defect_models": defect_models,
        "idx_to_component": idx_to_component,
        "transform": make_eval_transform(image_size),
    }


def get_threshold(component, route, metadata, mode, manual_threshold):
    if mode == "Final submission thresholds":
        if component in FINAL_COMPONENT_THRESHOLDS:
            return FINAL_COMPONENT_THRESHOLDS[component]
        return metadata["route_thresholds"][route]
    if mode == "Validation route thresholds":
        return metadata["route_thresholds"][route]
    return manual_threshold


@torch.no_grad()
def predict(image, bundle, threshold_mode, manual_threshold):
    metadata = bundle["metadata"]
    device = bundle["device"]
    x = bundle["transform"](image).unsqueeze(0).to(device)

    component_logits = bundle["component_model"](x)
    component_probs = torch.softmax(component_logits, dim=1).squeeze(0).cpu()
    component_idx = int(component_probs.argmax().item())
    component = bundle["idx_to_component"][component_idx]

    specialist_components = set(metadata.get("specialist_components", []))
    route = component if component in specialist_components else "global"

    defect_logits = bundle["defect_models"][route](x).squeeze(1)
    defect_score = float(torch.sigmoid(defect_logits).item())
    threshold = get_threshold(component, route, metadata, threshold_mode, manual_threshold)
    pred_label = int(defect_score >= threshold)

    component_rows = []
    for idx, probability in enumerate(component_probs.tolist()):
        name = bundle["idx_to_component"][idx]
        component_rows.append(
            {
                "component": COMPONENT_DISPLAY_NAMES.get(name, name),
                "probability": probability,
            }
        )

    return {
        "component": component,
        "route": route,
        "defect_score": defect_score,
        "threshold": threshold,
        "pred_label": pred_label,
        "component_probabilities": pd.DataFrame(component_rows).sort_values("probability", ascending=False),
    }


def main():
    st.set_page_config(page_title="Power-Line Asset Defect Classifier", page_icon="⚡", layout="centered")
    st.title("Power-Line Asset Defect Classifier")

    with st.sidebar:
        model_dir = st.text_input("Model directory", DEFAULT_MODEL_DIR)
        threshold_mode = st.selectbox(
            "Threshold mode",
            ["Final submission thresholds", "Validation route thresholds", "Manual threshold"],
        )
        manual_threshold = st.slider("Manual threshold", 0.0, 1.0, 0.50, 0.01)

    try:
        bundle = load_hybrid_model(model_dir)
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        st.code(
            "\n".join(
                [
                    f"{DEFAULT_MODEL_DIR}/metadata.json",
                    f"{DEFAULT_MODEL_DIR}/component_model.pt",
                    f"{DEFAULT_MODEL_DIR}/defect_global.pt",
                    f"{DEFAULT_MODEL_DIR}/defect_vari-grip.pt",
                    f"{DEFAULT_MODEL_DIR}/defect_glass-insulator.pt",
                    f"{DEFAULT_MODEL_DIR}/defect_yoke-suspension.pt",
                ]
            )
        )
        st.stop()

    uploaded = st.file_uploader("Upload an inspection image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a JPG or PNG image to run prediction.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption=uploaded.name, use_container_width=True)

    result = predict(image, bundle, threshold_mode, manual_threshold)
    component_name = COMPONENT_DISPLAY_NAMES.get(result["component"], result["component"])

    st.subheader("Prediction")
    col1, col2, col3 = st.columns(3)
    col1.metric("Component", component_name)
    col2.metric("Defect score", f"{result['defect_score']:.4f}")
    col3.metric("Threshold", f"{result['threshold']:.4f}")

    if result["pred_label"] == 1:
        st.error("DEFECTIVE / bad (1)")
    else:
        st.success("NORMAL / good (0)")

    st.write(f"Defect route: `{result['route']}`")

    with st.expander("Component probabilities"):
        probs = result["component_probabilities"].copy()
        probs["probability"] = probs["probability"].map(lambda value: f"{value:.4f}")
        st.dataframe(probs, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()

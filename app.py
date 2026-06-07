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
EXAMPLE_DIR = Path("app_examples")

COMPONENT_DISPLAY_NAMES = {
    "vari-grip": "vari-grip 可調式夾線條",
    "lightning-rod-suspension": "lightning-rod-suspension 避雷線懸吊裝置",
    "polymer-insulator-upper-shackle": "polymer-insulator-upper-shackle 絕緣子上部鉤環",
    "glass-insulator": "glass-insulator 玻璃絕緣子",
    "yoke-suspension": "yoke-suspension 聯軛懸吊組件",
}

COMPONENT_GUIDE = {
    "vari-grip": "用於夾持或固定線路的可調式夾具。缺失常見於零件脫落、鏽蝕或附著異物。",
    "lightning-rod-suspension": "避雷線相關的懸吊裝置。缺失常見於連接件異常、鏽蝕或異物附著。",
    "polymer-insulator-upper-shackle": "聚合物絕緣子上方的鉤環或連接件。缺失常見於金具損壞、鏽蝕或缺少零件。",
    "glass-insulator": "玻璃絕緣子串。缺失常見於破損、缺片、異物附著或鳥巢遮擋。",
    "yoke-suspension": "聯軛懸吊組件，用於多點連接與承力。缺失常見於鏽蝕、連接件缺失或鳥巢。",
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

    component_probabilities = {}
    component_rows = []
    for idx, probability in enumerate(component_probs.tolist()):
        name = bundle["idx_to_component"][idx]
        component_probabilities[name] = probability
        component_rows.append(
            {
                "component": COMPONENT_DISPLAY_NAMES.get(name, name),
                "probability": probability,
            }
        )

    return {
        "component": component,
        "component_display": COMPONENT_DISPLAY_NAMES.get(component, component),
        "route": route,
        "defect_score": defect_score,
        "threshold": threshold,
        "pred_label": pred_label,
        "prediction": "defective" if pred_label == 1 else "normal",
        "component_probabilities": pd.DataFrame(component_rows).sort_values("probability", ascending=False),
        "component_probability_values": component_probabilities,
    }


def result_to_row(filename, result):
    row = {
        "filename": filename,
        "component": result["component"],
        "component_display": result["component_display"],
        "defect_score": round(result["defect_score"], 6),
        "threshold": round(result["threshold"], 6),
        "prediction_label": result["pred_label"],
        "prediction": result["prediction"],
        "defect_route": result["route"],
    }
    for component, probability in result["component_probability_values"].items():
        row[f"prob_{component}"] = round(probability, 6)
    return row


def open_uploaded_image(uploaded_file):
    uploaded_file.seek(0)
    return Image.open(uploaded_file).convert("RGB")


def render_model_error(exc):
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


def render_prediction_badge(pred_label):
    if pred_label == 1:
        st.error("DEFECTIVE / bad (1)")
    else:
        st.success("NORMAL / good (0)")


def render_prediction_panel(results_df, first_result):
    st.subheader("Prediction results")
    st.caption("i  Defect score 大於等於 threshold 時，模型會輸出 defective / bad (1)。")

    metric_cols = st.columns(4)
    total = len(results_df)
    defective_count = int((results_df["prediction_label"] == 1).sum())
    normal_count = total - defective_count
    metric_cols[0].metric("Images", total, help="本次上傳並完成預測的圖片張數。")
    metric_cols[1].metric("Defective", defective_count, help="模型判斷為 defective / bad (1) 的圖片數。")
    metric_cols[2].metric("Normal", normal_count, help="模型判斷為 normal / good (0) 的圖片數。")
    metric_cols[3].metric("First score", f"{first_result['defect_score']:.4f}", help="第一張圖片的 defect score。")

    render_prediction_badge(first_result["pred_label"])
    st.write(f"First image component: `{first_result['component_display']}`")
    st.write(f"Defect route: `{first_result['route']}`")

    display_df = results_df[
        [
            "filename",
            "component_display",
            "defect_score",
            "threshold",
            "prediction_label",
            "prediction",
            "defect_route",
        ]
    ].copy()
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    csv_bytes = results_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Download prediction CSV",
        data=csv_bytes,
        file_name="streamlit_predictions.csv",
        mime="text/csv",
        help="下載本次上傳圖片的預測結果，包含檔名、設備類別、defect score、threshold 與預測 label。",
    )

    with st.expander("i  Component probabilities for the first image"):
        probs = first_result["component_probabilities"].copy()
        probs["probability"] = probs["probability"].map(lambda value: f"{value:.4f}")
        st.dataframe(probs, hide_index=True, use_container_width=True)


def render_upload_panel(uploaded_files):
    st.subheader("Upload images")
    st.caption("i  可以一次上傳多張 JPG / PNG。右側會顯示批次預測結果，並提供 CSV 下載。")

    if not uploaded_files:
        st.info("Upload one or more inspection images to run prediction.")
        return

    first_image = open_uploaded_image(uploaded_files[0])
    st.image(first_image, caption=f"Preview: {uploaded_files[0].name}", use_container_width=True)

    if len(uploaded_files) > 1:
        with st.expander(f"i  Uploaded image list ({len(uploaded_files)} files)"):
            for uploaded_file in uploaded_files:
                st.write(uploaded_file.name)


def render_guide():
    st.divider()
    st.header("使用說明與範例")

    guide_tab, examples_tab, model_tab = st.tabs(["操作說明", "五類 good / bad 範例", "模型流程"])

    with guide_tab:
        st.markdown(
            """
            1. 在左側上傳一張或多張巡檢圖片。
            2. 系統會先判斷圖片中的設備類別。
            3. 接著根據設備類別選擇 global defect classifier 或 specialist defect classifier。
            4. 若 defect score 大於等於 threshold，結果會被判斷為 `defective / bad (1)`；否則為 `normal / good (0)`。
            5. 預測完成後可以下載 CSV，方便整理批次圖片的結果。
            """
        )
        st.info("i  defective 通常代表照片中可能有缺失零件、鏽蝕、破損、鳥巢或其他異常附著物。")

    with examples_tab:
        for component, display_name in COMPONENT_DISPLAY_NAMES.items():
            st.subheader(display_name)
            st.caption(COMPONENT_GUIDE[component])
            cols = st.columns(2)
            for col, label in zip(cols, ["good", "bad"]):
                image_path = EXAMPLE_DIR / f"{component}-{label}.jpg"
                with col:
                    st.markdown(f"**{label.upper()} example**")
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.warning(f"Missing example image: {image_path}")

    with model_tab:
        st.markdown(
            """
            本 App 使用 final hybrid deep model：

            ```text
            input image
              -> EfficientNet-B0 component classifier
              -> 5-class equipment prediction
              -> choose defect route
              -> global or specialist defect classifier
              -> normal / defective
            ```

            Specialist defect classifiers 使用於資料量較足夠的類別：

            ```text
            vari-grip
            glass-insulator
            yoke-suspension
            ```

            `polymer-insulator-upper-shackle` 和 `lightning-rod-suspension` 的 bad 資料量較少，因此使用 global defect classifier，避免單一類別模型 overfit。
            """
        )


def main():
    st.set_page_config(
        page_title="Power-Line Asset Defect Classifier",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Power-Line Asset Defect Classifier")
    st.caption("Upload power-line inspection images and classify equipment type plus normal / defective status.")

    with st.sidebar:
        st.header("Settings")
        model_dir = st.text_input(
            "Model directory",
            DEFAULT_MODEL_DIR,
            help="模型權重資料夾。線上部署與本機預設都使用 hybrid_deep_models_p097。",
        )
        threshold_mode = st.selectbox(
            "Threshold mode",
            ["Final submission thresholds", "Validation route thresholds", "Manual threshold"],
            help="Final submission thresholds 是正式提交最佳版本使用的設定。",
        )
        manual_threshold = st.slider(
            "Manual threshold",
            0.0,
            1.0,
            0.50,
            0.01,
            help="只有在 Threshold mode 選 Manual threshold 時才會使用。",
        )
        st.divider()
        st.caption("i  線上版使用 CPU 推論，第一次開啟或大量上傳時可能需要稍等。")

    try:
        bundle = load_hybrid_model(model_dir)
    except Exception as exc:
        render_model_error(exc)
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload inspection images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="可以一次選取多張圖片。建議使用清楚包含目標設備的巡檢影像。",
        label_visibility="collapsed",
    )

    left_col, right_col = st.columns([0.92, 1.08], gap="large")

    with left_col:
        render_upload_panel(uploaded_files)

    with right_col:
        if uploaded_files:
            rows = []
            first_result = None
            progress = st.progress(0, text="Running predictions...")
            for index, uploaded_file in enumerate(uploaded_files, start=1):
                image = open_uploaded_image(uploaded_file)
                result = predict(image, bundle, threshold_mode, manual_threshold)
                if first_result is None:
                    first_result = result
                rows.append(result_to_row(uploaded_file.name, result))
                progress.progress(index / len(uploaded_files), text=f"Processed {index}/{len(uploaded_files)}")
            progress.empty()

            results_df = pd.DataFrame(rows)
            render_prediction_panel(results_df, first_result)
        else:
            st.subheader("Prediction results")
            st.info("Upload images on the left to see results here.")

    render_guide()


if __name__ == "__main__":
    main()

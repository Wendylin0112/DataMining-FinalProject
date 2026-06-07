import hashlib
import json
from io import BytesIO
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
PREVIEW_IMAGE_WIDTH = 460
PREVIEW_FULLSCREEN_MIN_WIDTH = 1000
EXAMPLE_IMAGE_WIDTH = 300

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


def get_model_dir_options():
    model_dirs = [
        path.name
        for path in Path(".").glob("hybrid_deep_models*")
        if path.is_dir() and (path / "metadata.json").exists()
    ]
    if DEFAULT_MODEL_DIR not in model_dirs:
        model_dirs.insert(0, DEFAULT_MODEL_DIR)
    return sorted(model_dirs, key=lambda name: (name != DEFAULT_MODEL_DIR, name))


def get_threshold(component, route, metadata, mode, manual_threshold):
    if mode == "Final submission thresholds":
        if component in FINAL_COMPONENT_THRESHOLDS:
            return FINAL_COMPONENT_THRESHOLDS[component]
        return metadata["route_thresholds"][route]
    if mode == "Validation thresholds":
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
    }
    for component, probability in result["component_probability_values"].items():
        row[f"prob_{component}"] = round(probability, 6)
    return row


def image_from_bytes(image_bytes):
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def upscale_for_fullscreen(image, min_width=PREVIEW_FULLSCREEN_MIN_WIDTH):
    if image.width >= min_width:
        return image
    scale = min_width / image.width
    target_size = (min_width, max(1, int(image.height * scale)))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return image.resize(target_size, resample)


def uploaded_name(uploaded_file):
    if isinstance(uploaded_file, dict):
        return uploaded_file["name"]
    return uploaded_file.name


def uploaded_bytes(uploaded_file):
    if isinstance(uploaded_file, dict):
        return uploaded_file["bytes"]
    return uploaded_file.getvalue()


def cache_key(uploaded_file, image_bytes, model_dir, threshold_mode, manual_threshold):
    digest = hashlib.md5(image_bytes).hexdigest()
    return (
        uploaded_name(uploaded_file),
        len(image_bytes),
        digest,
        model_dir,
        threshold_mode,
        round(float(manual_threshold), 4),
    )


def upload_id(filename, image_bytes):
    digest = hashlib.md5(image_bytes).hexdigest()[:12]
    return f"{filename}|{len(image_bytes)}|{digest}"


def upload_signature(uploaded_files):
    return tuple(upload_id(uploaded_name(uploaded_file), uploaded_bytes(uploaded_file)) for uploaded_file in uploaded_files)


def store_uploaded_files(uploaded_files):
    signature = upload_signature(uploaded_files)
    if st.session_state.get("uploaded_signature") == signature:
        return

    st.session_state.uploaded_images = [
        {
            "name": uploaded_name(uploaded_file),
            "bytes": uploaded_bytes(uploaded_file),
        }
        for uploaded_file in uploaded_files
    ]
    st.session_state.uploaded_signature = signature
    st.session_state.removed_upload_ids = set()
    st.session_state.selected_image_index = 0
    st.rerun()


def get_active_uploaded_files(uploaded_files):
    if "removed_upload_ids" not in st.session_state:
        st.session_state.removed_upload_ids = set()

    current_ids = {upload_id(uploaded_name(uploaded_file), uploaded_bytes(uploaded_file)) for uploaded_file in uploaded_files}
    st.session_state.removed_upload_ids = st.session_state.removed_upload_ids.intersection(current_ids)

    return [
        uploaded_file
        for uploaded_file in uploaded_files
        if upload_id(uploaded_name(uploaded_file), uploaded_bytes(uploaded_file)) not in st.session_state.removed_upload_ids
    ]


def predict_uploaded_files(uploaded_files, bundle, model_dir, threshold_mode, manual_threshold):
    if "prediction_cache" not in st.session_state:
        st.session_state.prediction_cache = {}

    items = []
    progress = st.progress(0, text="Running predictions...")
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        image_bytes = uploaded_bytes(uploaded_file)
        filename = uploaded_name(uploaded_file)
        current_upload_id = upload_id(filename, image_bytes)
        key = cache_key(uploaded_file, image_bytes, model_dir, threshold_mode, manual_threshold)
        if key not in st.session_state.prediction_cache:
            image = image_from_bytes(image_bytes)
            result = predict(image, bundle, threshold_mode, manual_threshold)
            st.session_state.prediction_cache[key] = {
                "result": result,
                "row": result_to_row(filename, result),
            }

        cached = st.session_state.prediction_cache[key]
        items.append(
            {
                "upload_id": current_upload_id,
                "filename": filename,
                "image_bytes": image_bytes,
                "result": cached["result"],
                "row": cached["row"],
            }
        )
        progress.progress(index / len(uploaded_files), text=f"Processed {index}/{len(uploaded_files)}")
    progress.empty()
    return items, pd.DataFrame([item["row"] for item in items])


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


def clamp_selected_index(item_count):
    if "selected_image_index" not in st.session_state:
        st.session_state.selected_image_index = 0
    if item_count <= 0:
        st.session_state.selected_image_index = 0
    else:
        st.session_state.selected_image_index = min(st.session_state.selected_image_index, item_count - 1)


def reset_upload_state():
    st.session_state.uploader_version = st.session_state.get("uploader_version", 0) + 1
    st.session_state.selected_image_index = 0
    st.session_state.removed_upload_ids = set()
    st.session_state.uploaded_images = []
    st.session_state.uploaded_signature = ()


def render_image_navigation(items):
    clamp_selected_index(len(items))
    filenames = [item["filename"] for item in items]

    prev_col, select_col, next_col = st.columns([0.09, 0.82, 0.09])
    with prev_col:
        if st.button("←", use_container_width=True, disabled=st.session_state.selected_image_index <= 0, help="Previous image"):
            st.session_state.selected_image_index -= 1
    with next_col:
        if st.button(
            "→",
            use_container_width=True,
            disabled=st.session_state.selected_image_index >= len(items) - 1,
            help="Next image",
        ):
            st.session_state.selected_image_index += 1

    clamp_selected_index(len(items))
    with select_col:
        selected_index = st.selectbox(
            "Preview image",
            options=list(range(len(items))),
            format_func=lambda idx: f"{idx + 1}. {filenames[idx]}",
            index=st.session_state.selected_image_index,
            help="也可以用左右箭頭按鈕切換目前預覽的照片。",
        )
    st.session_state.selected_image_index = selected_index
    return selected_index


def render_results_table(results_df, items):
    st.caption("在 remove 欄位勾選圖片後，按下 Remove checked images 可從目前批次結果中移除。")

    display_df = results_df[
        [
            "filename",
            "component_display",
            "defect_score",
            "threshold",
            "prediction_label",
            "prediction",
        ]
    ].copy()
    display_df.insert(0, "remove", False)

    edited_df = st.data_editor(
        display_df,
        hide_index=True,
        use_container_width=True,
        disabled=[
            "filename",
            "component_display",
            "defect_score",
            "threshold",
            "prediction_label",
            "prediction",
        ],
        column_config={
            "remove": st.column_config.CheckboxColumn(
                "remove",
                help="勾選後按下方按鈕，可從目前批次中移除這張照片。",
            )
        },
        key="prediction_remove_table",
    )

    checked_positions = edited_df.index[edited_df["remove"]].tolist()
    remove_disabled = len(checked_positions) == 0
    if st.button("Remove checked images", disabled=remove_disabled, use_container_width=True):
        for position in checked_positions:
            st.session_state.removed_upload_ids.add(items[position]["upload_id"])
        st.session_state.selected_image_index = 0
        st.rerun()


def render_prediction_panel(results_df, items, selected_result):
    st.subheader("Prediction results")
    st.caption("Defect score 大於等於 threshold 時，模型會輸出 defective / bad (1)。")

    metric_cols = st.columns(4)
    total = len(results_df)
    defective_count = int((results_df["prediction_label"] == 1).sum())
    normal_count = total - defective_count
    metric_cols[0].metric("Images", total, help="本次上傳並完成預測的圖片張數。")
    metric_cols[1].metric("Defective", defective_count, help="模型判斷為 defective / bad (1) 的圖片數。")
    metric_cols[2].metric("Normal", normal_count, help="模型判斷為 normal / good (0) 的圖片數。")
    metric_cols[3].metric("Selected score", f"{selected_result['defect_score']:.4f}", help="目前選取圖片的 defect score。")

    render_prediction_badge(selected_result["pred_label"])
    st.write(f"Selected component: `{selected_result['component_display']}`")

    render_results_table(results_df, items)

    csv_bytes = results_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Download prediction CSV",
        data=csv_bytes,
        file_name="streamlit_predictions.csv",
        mime="text/csv",
        help="下載本次上傳圖片的預測結果，包含檔名、設備類別、defect score、threshold 與預測 label。",
    )

    with st.expander("Component probabilities for the selected image"):
        probs = selected_result["component_probabilities"].copy()
        probs["probability"] = probs["probability"].map(lambda value: f"{value:.4f}")
        st.dataframe(probs, hide_index=True, use_container_width=True)


def render_upload_panel(uploaded_files, selected_item=None, selected_result=None):
    if not uploaded_files:
        st.info("Upload one or more inspection images to run prediction.")
        return

    if selected_item is None:
        st.info("Preparing preview...")
        return

    selected_image = upscale_for_fullscreen(image_from_bytes(selected_item["image_bytes"]))
    st.image(selected_image, caption=f"Preview: {selected_item['filename']}", width=PREVIEW_IMAGE_WIDTH)

    if selected_result is not None:
        st.write(f"Component: `{selected_result['component_display']}`")
        st.write(f"Prediction: `{selected_result['prediction']}`")
        st.write(f"Defect score: `{selected_result['defect_score']:.4f}`")


def render_threshold_explanation(metadata):
    st.sidebar.divider()
    st.sidebar.subheader("Threshold mode 說明")
    st.sidebar.markdown(
        """
        **Final submission thresholds**

        使用正式提交最佳版本的 threshold。大多數模型分支使用 validation tuning 的 threshold，並針對部分設備類別做 component-level 覆寫，以在 precision 維持 0.90 以上時提高 recall。

        **Validation thresholds**

        使用訓練時在 validation set 上搜尋出的 threshold。

        **Manual threshold**

        使用你手動設定的單一 threshold。數值越低越容易判斷為 defective，通常 recall 會上升，但 precision 可能下降。
        """
    )

    final_rows = []
    for component, threshold in FINAL_COMPONENT_THRESHOLDS.items():
        final_rows.append(
            {
                "component": COMPONENT_DISPLAY_NAMES.get(component, component),
                "threshold": threshold,
                "source": "final component override",
            }
        )
    st.sidebar.dataframe(pd.DataFrame(final_rows), hide_index=True, use_container_width=True)

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
        st.info("defective 通常代表照片中可能有缺失零件、鏽蝕、破損、鳥巢或其他異常附著物。")

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
                        st.image(str(image_path), width=EXAMPLE_IMAGE_WIDTH)
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
              -> choose defect classifier
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
    st.markdown(
        """
        <style>
        div[data-testid="stFileUploaderFile"] {
            display: none;
        }
        div[data-testid="stImage"] img {
            max-width: 100%;
            height: auto;
        }
        div[data-testid="stButton"] button {
            min-height: 3rem;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if "uploader_version" not in st.session_state:
        st.session_state.uploader_version = 0

    st.title("Power-Line Asset Defect Classifier")
    st.caption("Upload power-line inspection images and classify equipment type plus normal / defective status.")

    with st.sidebar:
        st.header("Settings")
        model_options = get_model_dir_options()
        model_dir = st.selectbox(
            "Model directory",
            model_options,
            index=model_options.index(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR in model_options else 0,
            help="選擇要使用的模型權重資料夾。線上部署預設使用 final model：hybrid_deep_models_p097。",
        )
        threshold_mode = st.selectbox(
            "Threshold mode",
            ["Final submission thresholds", "Validation thresholds", "Manual threshold"],
            help="選擇 defect score 要使用哪一組 threshold 轉換成 normal / defective。",
        )
        manual_threshold = 0.50
        if threshold_mode == "Manual threshold":
            manual_threshold = st.slider(
                "Manual threshold",
                0.0,
                1.0,
                0.50,
                0.01,
                help="數值越低越容易判斷為 defective；數值越高越保守。",
            )
        st.divider()
        st.caption("線上版使用 CPU 推論，第一次開啟或大量上傳時可能需要稍等。")

    try:
        bundle = load_hybrid_model(model_dir)
    except Exception as exc:
        render_model_error(exc)
        st.stop()

    render_threshold_explanation(bundle["metadata"])

    stored_uploaded_files = st.session_state.get("uploaded_images", [])

    left_col, right_col = st.columns([0.72, 1.28], gap="large")
    with left_col:
        if stored_uploaded_files:
            upload_title_col, reset_col = st.columns([0.62, 0.38])
            with upload_title_col:
                st.subheader("Upload images")
            with reset_col:
                st.write("")
                if st.button("重新上傳", use_container_width=True, help="清除目前上傳清單與移除狀態，重新選擇圖片。"):
                    reset_upload_state()
                    st.rerun()
        else:
            st.subheader("Upload images")
        st.caption("可以一次上傳多張 JPG / PNG。右側會顯示批次預測結果，並提供 CSV 下載。")
        if stored_uploaded_files:
            uploaded_files = stored_uploaded_files
        else:
            raw_uploaded_files = st.file_uploader(
                "Upload inspection images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="可以一次選取多張圖片。建議使用清楚包含目標設備的巡檢影像。",
                label_visibility="collapsed",
                key=f"inspection_uploads_{st.session_state.uploader_version}",
            )
            if raw_uploaded_files:
                store_uploaded_files(raw_uploaded_files)
            uploaded_files = []
        preview_slot = st.empty()

    active_uploaded_files = get_active_uploaded_files(uploaded_files) if uploaded_files else []
    selected_item = None
    selected_result = None
    items = []
    results_df = pd.DataFrame()

    if active_uploaded_files:
        items, results_df = predict_uploaded_files(
            active_uploaded_files,
            bundle,
            model_dir,
            threshold_mode,
            manual_threshold,
        )
        clamp_selected_index(len(items))

    with preview_slot.container():
        if active_uploaded_files:
            selected_index = render_image_navigation(items)
            selected_item = items[selected_index]
            selected_result = selected_item["result"]
        render_upload_panel(active_uploaded_files, selected_item, selected_result)

    with right_col:
        if active_uploaded_files:
            render_prediction_panel(results_df, items, selected_result)
        elif uploaded_files:
            st.subheader("Prediction results")
            st.info("All uploaded images were removed from the current batch. Upload new images or clear the uploader to start over.")
        else:
            st.subheader("Prediction results")
            st.info("Upload images on the left to see results here.")

    render_guide()


if __name__ == "__main__":
    main()

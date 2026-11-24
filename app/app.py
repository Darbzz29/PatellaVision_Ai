"""Single-slice review interface for the experimental knee MRI model."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
import streamlit as st
import torch
import torch.nn.functional as functional
from PIL import Image
from pydicom.errors import InvalidDicomError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import normalize_mri_array
from src.model import UNet

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_unet.pt"
TEST_SUMMARY_PATH = PROJECT_ROOT / "results" / "test_metrics_summary.json"

st.set_page_config(page_title="PatellaVision AI", layout="wide")


@st.cache_resource
def load_model(checkpoint_path: str) -> tuple[UNet, int, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet(
        in_channels=int(checkpoint["in_channels"]),
        out_channels=int(checkpoint["out_channels"]),
        base_channels=int(checkpoint["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, int(checkpoint["image_size"]), device


def prepare_dicom(
    uploaded_bytes: bytes,
    image_size: int,
) -> tuple[np.ndarray, torch.Tensor, dict[str, str]]:
    dataset = pydicom.dcmread(io.BytesIO(uploaded_bytes))
    pixel_array = dataset.pixel_array
    normalized = normalize_mri_array(
        pixel_array,
        dataset.get("PhotometricInterpretation", "MONOCHROME2"),
    )
    image = torch.from_numpy(normalized)[None, None, :, :]
    metadata = {
        "Modality": str(dataset.get("Modality", "Not provided")),
        "Image matrix": f"{pixel_array.shape[1]} x {pixel_array.shape[0]} pixels",
        "Pixel spacing": _format_pair(dataset.get("PixelSpacing"), "mm"),
        "Slice thickness": _format_measurement(
            dataset.get("SliceThickness"), "mm"
        ),
        "Magnetic field strength": _format_measurement(
            dataset.get("MagneticFieldStrength"), "T"
        ),
        "Repetition time (TR)": _format_measurement(
            dataset.get("RepetitionTime"), "ms"
        ),
        "Echo time (TE)": _format_measurement(dataset.get("EchoTime"), "ms"),
    }
    model_input = functional.interpolate(
        image,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return normalized, model_input, metadata


def sort_dicom_series(
    uploaded_files: list[tuple[str, bytes]],
) -> tuple[list[tuple[str, bytes]], str]:
    if not uploaded_files:
        raise ValueError("Upload at least one DICOM slice")

    slices = list(uploaded_files)
    datasets = [
        pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True)
        for _, content in slices
    ]
    if any(str(dataset.get("Modality", "")).upper() != "MR" for dataset in datasets):
        raise ValueError("Every uploaded file must be an MR image")

    series_uids = {
        str(dataset.get("SeriesInstanceUID", "")).strip() for dataset in datasets
    }
    if len(series_uids) != 1 or "" in series_uids:
        raise ValueError("Upload slices from one DICOM series with a Series Instance UID")

    dimensions = {
        (int(dataset.Rows), int(dataset.Columns)) for dataset in datasets
    }
    if len(dimensions) != 1:
        raise ValueError("All slices in the series must have the same image dimensions")

    orientations = [dataset.get("ImageOrientationPatient") for dataset in datasets]
    positions = [dataset.get("ImagePositionPatient") for dataset in datasets]
    if all(value is not None for value in orientations + positions):
        try:
            orientation_array = np.asarray(orientations, dtype=np.float64)
            if orientation_array.shape != (len(slices), 6) or not np.allclose(
                orientation_array,
                orientation_array[0],
                atol=1e-4,
            ):
                raise ValueError("Slices have inconsistent image orientation")
            normal = np.cross(orientation_array[0, :3], orientation_array[0, 3:])
            normal_length = np.linalg.norm(normal)
            if not np.isfinite(normal_length) or normal_length <= 0:
                raise ValueError("DICOM image orientation is invalid")
            normal /= normal_length
            coordinates = [
                float(np.dot(np.asarray(position, dtype=np.float64), normal))
                for position in positions
            ]
            if not np.all(np.isfinite(coordinates)):
                raise ValueError("DICOM slice positions are invalid")
            order = np.argsort(coordinates).tolist()
            ordered_coordinates = [coordinates[index] for index in order]
            if any(
                np.isclose(first, second, atol=1e-4)
                for first, second in zip(
                    ordered_coordinates,
                    ordered_coordinates[1:],
                )
            ):
                raise ValueError("The series contains duplicate slice positions")
            order_description = "physical image position"
        except (TypeError, ValueError) as error:
            raise ValueError(f"Could not reliably order this DICOM series: {error}") from error
    else:
        try:
            instance_numbers = [int(dataset.InstanceNumber) for dataset in datasets]
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "Slices need image-position tags or unique instance numbers for ordering"
            ) from error
        if len(set(instance_numbers)) != len(instance_numbers):
            raise ValueError("The series contains duplicate DICOM instance numbers")
        order = np.argsort(instance_numbers).tolist()
        order_description = "DICOM instance number"

    return [slices[index] for index in order], order_description


def _format_measurement(value: object, unit: str) -> str:
    if value is None:
        return "Not provided"
    try:
        return f"{float(value):g} {unit}"
    except (TypeError, ValueError):
        return "Not provided"


def _format_pair(value: object, unit: str) -> str:
    if value is None:
        return "Not provided"
    try:
        first, second = value
        return f"{float(first):g} x {float(second):g} {unit}"
    except (TypeError, ValueError):
        return "Not provided"


def make_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    opacity: float = 0.45,
) -> np.ndarray:
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    tint = np.asarray([1.0, 0.15, 0.1], dtype=np.float32)
    rgb[mask] = (1 - opacity) * rgb[mask] + opacity * tint
    return np.clip(rgb, 0, 1)


def to_png_bytes(image: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def make_review_log_csv(rows: list[tuple[int, str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("slice_number", "research_review_status", "research_note"))
    for slice_number, status, note in rows:
        safe_status = _safe_csv_value(status)
        safe_note = _safe_csv_value(note)
        writer.writerow((slice_number, safe_status, safe_note))
    return output.getvalue().encode("utf-8")


def summarize_mask(mask: np.ndarray) -> dict[str, float | int | tuple[int, int, int, int]]:
    if mask.size == 0:
        return {"foreground_ratio": 0.0, "area_pixels": 0, "bbox": (0, 0, 0, 0)}

    foreground = mask.astype(bool)
    area_pixels = int(foreground.sum())
    foreground_ratio = float(area_pixels / foreground.size)
    ys, xs = np.where(foreground)
    if xs.size == 0:
        bbox = (0, 0, 0, 0)
    else:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return {"foreground_ratio": foreground_ratio, "area_pixels": area_pixels, "bbox": bbox}


def patient_friendly_summary(
    slice_index: int,
    total_slices: int,
    mask_summary: dict[str, float | int | tuple[int, int, int, int]],
    review_status: str,
    review_note: str,
) -> str:
    mask_ratio = float(mask_summary["foreground_ratio"]) * 100.0
    area_pixels = int(mask_summary["area_pixels"])
    left, top, right, bottom = mask_summary["bbox"]
    bbox_width = max(0, right - left + 1)
    bbox_height = max(0, bottom - top + 1)
    summary_lines = [
        "PatellaVision AI — patient-friendly summary",
        "",
        f"This is slice {slice_index + 1} of {total_slices} in the MRI series.",
        (
            f"The highlighted region covers about {mask_ratio:.1f}% of the image, "
            f"which corresponds to approximately {area_pixels} foreground pixels."
        ),
        (
            "In plain language, the red area is the model's estimate of the most "
            "important bone-related region on this slice. It is useful for discussing "
            "the anatomy and the slice-by-slice pattern, but it is not a final diagnosis."
        ),
        (
            f"The approximate local focus spans {bbox_width} by {bbox_height} pixels in this slice. "
            "This can help clinicians compare the region of interest across adjacent slices."
        ),
        f"Review status: {review_status}",
        f"Clinical note: {review_note or 'No additional note recorded.'}",
        "",
        "This output is research support only and should be reviewed against the original imaging and clinical context.",
    ]
    return "\n".join(summary_lines)


def build_patient_education_note(
    slice_index: int,
    total_slices: int,
    mask_ratio: float,
    bbox_width: int,
    bbox_height: int,
    review_status: str,
) -> str:
    return (
        "Patient education note template\n"
        "\n"
        f"This MRI slice ({slice_index + 1} of {total_slices}) highlights the main area of interest in red. "
        f"The model estimates that roughly {mask_ratio:.1f}% of this image is involved in the highlighted region, "
        f"with a local focus of approximately {bbox_width} by {bbox_height} pixels.\n\n"
        "This helps the care team discuss where the key anatomy is located across adjacent slices. "
        "It is not a diagnosis, a measurement, or a guarantee of a surgical finding. The original images and clinical review remain essential.\n\n"
        f"Current review status: {review_status}."
    )


def build_case_checklist_text(checklist: dict[str, bool]) -> str:
    items = []
    for label, checked in checklist.items():
        status = "Yes" if checked else "No"
        items.append(f"- {label}: {status}")
    return "\n".join(items) if items else "- No checklist items recorded."


def build_surgeon_case_report(
    slice_index: int,
    total_slices: int,
    mask_ratio: float,
    bbox_width: int,
    bbox_height: int,
    review_status: str,
    review_note: str,
    checklist: dict[str, bool],
    surgeon_summary: str,
) -> str:
    lines = [
        "PatellaVision AI — surgical case report",
        "",
        f"Slice: {slice_index + 1} of {total_slices}",
        f"Highlighted region: {mask_ratio:.1f}% of image area",
        f"Approximate ROI size: {bbox_width} x {bbox_height} pixels",
        f"Review status: {review_status}",
        "",
        "Pre-operative checklist:",
        build_case_checklist_text(checklist),
        "",
        "Surgeon notes:",
        surgeon_summary or "No surgeon notes recorded.",
        "",
        "Clinical context:",
        review_note or "No additional review note recorded.",
        "",
        "Important safety note:",
        "This tool provides a research-only overlay for discussion support and must not replace direct review of the original MR images, clinical history, or surgical judgment.",
    ]
    return "\n".join(lines)


def _safe_csv_value(value: str) -> str:
    if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"
    return value


def show_test_metrics() -> None:
    if not TEST_SUMMARY_PATH.is_file():
        return
    metrics = json.loads(TEST_SUMMARY_PATH.read_text(encoding="utf-8"))
    with st.expander("Research-only evaluation (not this slice)"):
        st.caption(
            "These metrics are from the project's 43-slice contiguous holdout "
            "in the available single volume. They are not validation results "
            "for the uploaded image."
        )
        columns = st.columns(2)
        columns[0].metric("Mean Dice", f"{metrics['mean_per_slice_dice']:.4f}")
        columns[1].metric("Mean IoU", f"{metrics['mean_per_slice_iou']:.4f}")


st.title("PatellaVision AI")
st.write("MRI series review workspace")
st.warning(
    "Research prototype only - not clinically validated or a medical device. "
    "Do not use its output for diagnosis, treatment, or clinical decisions. "
    "A qualified clinician must review the source images independently."
)
st.markdown(
    "Upload the **DICOM slices from one MR series** to browse the model's "
    "experimental binary mask alongside each slice and prepare a more useful "
    "shared explanation for clinicians and surgeons."
)
guide_steps = {
    "1. Load series": (
        "Upload every slice from one MR series. The app checks that the files "
        "share a series UID and orders them using DICOM position when available."
    ),
    "2. Verify context": (
        "Confirm patient side, orientation, and scan context in your validated "
        "imaging viewer. This app preserves pixel-array orientation and cannot "
        "confirm laterality."
    ),
    "3. Review predictions": (
        "Navigate through adjacent slices and compare the MRI, red overlay, "
        "and binary mask. Red marks predicted foreground only; it is not a "
        "diagnosis or a verified anatomical boundary."
    ),
    "4. Export for research": (
        "Export the selected slice's PNG mask or overlay for teaching or "
        "research review, and save a review log. These files are not "
        "DICOM-SEG, a clinical report, or a surgical plan."
    ),
    "5. Prepare case discussion": (
        "Review the case with a pre-op checklist, surgeon notes, and patient-friendly "
        "language so the anatomy can be explained more clearly and documented for team discussion."
    ),
}
guide_step = st.radio(
    "Guided review steps",
    tuple(guide_steps),
    horizontal=True,
)
st.info(guide_steps[guide_step])
st.caption("This guide explains the prototype workflow, not clinical interpretation.")
with st.sidebar:
    st.subheader("How this supports review")
    st.markdown(
        "- Browse an MR series in spatial order\n"
        "- Inspect a model-highlighted region on each selected slice\n"
        "- Flag slices for manual correction and keep research notes\n"
        "- Export a research mask, overlay, and review log\n"
        "- The mask merges annotated foreground; it does not identify each bone"
    )
    st.subheader("Limits for surgical planning")
    st.markdown(
        "This prototype cannot assess alignment, diagnose an injury, measure "
        "surgical parameters, or recommend a procedure. Its model was tested "
        "on slices from one volume, not on independent patients."
    )
    st.info(
        "Use only de-identified data or data in an environment approved by "
        "your organization. Uploads are processed in session memory and are "
        "not written to project files."
    )

if not CHECKPOINT_PATH.is_file():
    st.error(f"Trained model checkpoint not found: {CHECKPOINT_PATH}")
    st.stop()

uploaded_files = st.file_uploader(
    "Choose DICOM slices from one MR series",
    type=["dcm"],
    accept_multiple_files=True,
    help="Select all slices from one series. Mixed series are rejected.",
)
if uploaded_files:
    try:
        ordered_slices, order_description = sort_dicom_series(
            [
                (uploaded_file.name, uploaded_file.getvalue())
                for uploaded_file in uploaded_files
            ]
        )
    except (
        InvalidDicomError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        st.error(f"Could not read this DICOM series: {error}")
    else:
        series_token = hashlib.sha256(
            "\0".join(str(uploaded_file.file_id) for uploaded_file in uploaded_files).encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        if st.session_state.get("_research_series_token") != series_token:
            for key in tuple(st.session_state):
                if key.startswith(("research_review_status_", "research_review_note_")):
                    del st.session_state[key]
                if key.startswith(("surgical_checklist_", "surgeon_summary_", "patient_note_")):
                    del st.session_state[key]
            st.session_state["_research_series_token"] = series_token
            st.session_state["_research_review_records"] = {}
            st.session_state["_case_context"] = {
                "checklist": {},
                "surgeon_summary": "",
                "patient_note": "",
            }

        st.subheader("Review a slice")
        if len(ordered_slices) == 1:
            slice_index = 0
        else:
            slice_index = st.slider(
                "Slice in series",
                min_value=1,
                max_value=len(ordered_slices),
                value=1,
            )
            slice_index -= 1
        slice_name, slice_bytes = ordered_slices[slice_index]
        st.caption(
            f"Slice {slice_index + 1} of {len(ordered_slices)} - "
            f"sorted by {order_description} - {slice_name}"
        )

        model, image_size, device = load_model(str(CHECKPOINT_PATH))
        try:
            original_image, model_input, metadata = prepare_dicom(
                slice_bytes,
                image_size,
            )
        except (
            InvalidDicomError,
            AttributeError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            st.error(f"Could not decode this DICOM slice: {error}")
        else:
            with torch.inference_mode():
                logits = model(model_input.to(device))
                prediction = (torch.sigmoid(logits) >= 0.5).float()
                prediction = functional.interpolate(
                    prediction,
                    size=original_image.shape,
                    mode="nearest",
                )[0, 0].cpu().numpy().astype(bool)

            st.caption(
                "The display uses the raw pixel-array orientation. Confirm "
                "orientation and anatomy in a validated imaging viewer."
            )
            metadata_items = list(metadata.items())
            for start in (0, 4):
                detail_columns = st.columns(4)
                for column, (label, value) in zip(
                    detail_columns, metadata_items[start : start + 4]
                ):
                    column.metric(label, value)

            mask_summary = summarize_mask(prediction)
            mask_ratio = float(mask_summary["foreground_ratio"]) * 100.0
            bbox = mask_summary["bbox"]
            bbox_width = max(0, bbox[2] - bbox[0] + 1)
            bbox_height = max(0, bbox[3] - bbox[1] + 1)
            st.subheader("Clinical explanation and surgical reference")
            explanation_columns = st.columns(3)
            explanation_columns[0].metric("Foreground share", f"{mask_ratio:.1f}%")
            explanation_columns[1].metric("ROI width", f"{bbox_width}px")
            explanation_columns[2].metric("ROI height", f"{bbox_height}px")

            with st.expander("Plain-language explanation for patients and clinicians", expanded=True):
                st.markdown(
                    "This slice shows the main foreground region highlighted in red. "
                    "The model is estimating the dominant bone-related area on this MRI slice. "
                    "This is useful for explaining where the region of interest sits and how it changes "
                    "from one slice to the next, but it is not a final diagnosis or a surgical measurement."
                )
                st.markdown(
                    f"On this slice, about {mask_ratio:.1f}% of the image area is highlighted, "
                    f"and the strongest local focus spans roughly {bbox_width} by {bbox_height} pixels. "
                    "For patient communication, this can help explain the approximate location of the key anatomy "
                    "without making claims about diagnosis, treatment, or procedural planning."
                )

            controls = st.columns([2, 1])
            view_mode = controls[0].radio(
                "Display",
                ("Side-by-side", "Overlay", "Mask only"),
                horizontal=True,
            )
            opacity = controls[1].slider(
                "Overlay opacity",
                min_value=0.1,
                max_value=0.9,
                value=0.45,
                step=0.05,
                disabled=view_mode == "Mask only",
            )

            prediction_u8 = prediction.astype(np.uint8) * 255
            overlay = make_overlay(original_image, prediction, opacity)
            if view_mode == "Side-by-side":
                image_column, overlay_column, mask_column = st.columns(3)
                image_column.image(
                    original_image,
                    caption="Normalized MRI",
                    clamp=True,
                    width="stretch",
                )
                overlay_column.image(
                    overlay,
                    caption="MRI with predicted mask",
                    clamp=True,
                    width="stretch",
                )
                mask_column.image(
                    prediction_u8,
                    caption="Binary predicted mask",
                    width="stretch",
                )
            elif view_mode == "Overlay":
                st.image(
                    overlay,
                    caption="MRI with predicted mask",
                    clamp=True,
                    width="stretch",
                )
            else:
                st.image(
                    prediction_u8,
                    caption="Binary predicted mask",
                    width="stretch",
                )

            st.info(
                "Inspect the overlay for alignment and segmentation errors. "
                "The mask is an experimental merged-foreground output, not a "
                "verified anatomical contour or surgical measurement."
            )
            with st.expander("What this can show"):
                st.write(
                    "The highlighted pixels are the model's predicted binary "
                    "foreground. The training annotations combine bones into "
                    "one class, so this view cannot identify the patella "
                    "separately or assess a surgical finding."
                )
                st.write(
                    "Series navigation can support teaching or research review "
                    "of predictions across adjacent slices. No clinical benefit "
                    "or time saving has been established."
                )
            st.subheader("Research review record")
            st.caption(
                "Flag the model output for follow-up; this does not edit the "
                "mask or approve it for clinical use."
            )
            status_options = (
                "Not reviewed",
                "No obvious discrepancy (research only)",
                "Needs manual correction",
                "Unable to assess",
            )
            review_records = st.session_state["_research_review_records"]
            saved_record = review_records.get(slice_index, {})
            saved_status = saved_record.get("status", status_options[0])
            saved_note = saved_record.get("note", "")
            form_key = f"research_review_form_{series_token}_{slice_index}"
            with st.form(form_key):
                review_status = st.selectbox(
                    "Research review status",
                    status_options,
                    index=status_options.index(saved_status),
                    key=f"research_review_status_{series_token}_{slice_index}",
                )
                review_note = st.text_area(
                    "Research note (do not enter identifying information)",
                    value=saved_note,
                    key=f"research_review_note_{series_token}_{slice_index}",
                )
                save_review = st.form_submit_button("Save slice review")
            if save_review:
                review_records[slice_index] = {
                    "status": review_status,
                    "note": review_note,
                }
                st.success("Research review saved for this session.")

            st.subheader("Pre-op planning checklist")
            case_context = st.session_state["_case_context"]
            checklist_defaults = {
                "Series reviewed in full": case_context["checklist"].get("Series reviewed in full", False),
                "Adjacent slices reviewed": case_context["checklist"].get("Adjacent slices reviewed", False),
                "Orientation confirmed in primary viewer": case_context["checklist"].get("Orientation confirmed in primary viewer", False),
                "Clinical context reviewed": case_context["checklist"].get("Clinical context reviewed", False),
                "Potential surgical relevance discussed": case_context["checklist"].get("Potential surgical relevance discussed", False),
                "Further imaging review needed": case_context["checklist"].get("Further imaging review needed", False),
            }
            with st.form(f"surgical_case_form_{series_token}_{slice_index}"):
                checklist_values = {}
                checklist_columns = st.columns(2)
                for index, (label, default_value) in enumerate(checklist_defaults.items()):
                    column = checklist_columns[index % 2]
                    checklist_values[label] = column.checkbox(
                        label,
                        value=default_value,
                        key=f"surgical_checklist_{series_token}_{slice_index}_{index}",
                    )
                surgeon_summary = st.text_area(
                    "Surgeon summary / operative discussion",
                    value=case_context["surgeon_summary"],
                    key=f"surgeon_summary_{series_token}_{slice_index}",
                    help="Use this for concise case discussion, planned views, or surgical considerations.",
                )
                save_case = st.form_submit_button("Save case context")
            if save_case:
                case_context["checklist"] = checklist_values
                case_context["surgeon_summary"] = surgeon_summary
                st.session_state["_case_context"] = case_context
                st.success("Surgical planning notes saved for this session.")

            st.subheader("Patient education note template")
            patient_note = build_patient_education_note(
                slice_index,
                len(ordered_slices),
                mask_ratio,
                bbox_width,
                bbox_height,
                review_records.get(slice_index, {}).get("status", status_options[0]),
            )
            editable_patient_note = st.text_area(
                "Editable note for discussion with the patient",
                value=case_context.get("patient_note") or patient_note,
                key=f"patient_note_{series_token}_{slice_index}",
                help="This is a template for explaining the highlighted area in simpler language without making a diagnosis.",
            )
            if st.button("Save patient note", key=f"save_patient_note_{series_token}_{slice_index}"):
                case_context["patient_note"] = editable_patient_note
                st.session_state["_case_context"] = case_context
                st.success("Patient education note updated.")

            st.caption(
                "Review notes stay in this session and are cleared when you "
                "switch series. Download the log before switching."
            )
            st.caption(
                f"Source matrix: {original_image.shape[1]} x {original_image.shape[0]} | "
                f"Model input: {image_size} x {image_size} | "
                f"Inference device: {device}"
            )
            review_rows = [
                (
                    index + 1,
                    review_records.get(index, {}).get("status", status_options[0]),
                    review_records.get(index, {}).get("note", ""),
                )
                for index in range(len(ordered_slices))
            ]
            patient_summary = patient_friendly_summary(
                slice_index,
                len(ordered_slices),
                mask_summary,
                review_records.get(slice_index, {}).get("status", status_options[0]),
                review_records.get(slice_index, {}).get("note", ""),
            )
            case_report = build_surgeon_case_report(
                slice_index,
                len(ordered_slices),
                mask_ratio,
                bbox_width,
                bbox_height,
                review_records.get(slice_index, {}).get("status", status_options[0]),
                review_records.get(slice_index, {}).get("note", ""),
                case_context.get("checklist", {}),
                case_context.get("surgeon_summary", ""),
            )
            downloads = st.columns(5)
            downloads[0].download_button(
                "Download research mask (PNG)",
                data=to_png_bytes(prediction_u8),
                file_name=f"patellavision_research_mask_slice_{slice_index + 1:04d}.png",
                mime="image/png",
            )
            downloads[1].download_button(
                "Download review overlay (PNG)",
                data=to_png_bytes(np.round(overlay * 255).astype(np.uint8)),
                file_name=f"patellavision_research_overlay_slice_{slice_index + 1:04d}.png",
                mime="image/png",
            )
            downloads[2].download_button(
                "Download patient summary (TXT)",
                data=patient_summary.encode("utf-8"),
                file_name=f"patellavision_patient_summary_slice_{slice_index + 1:04d}.txt",
                mime="text/plain",
            )
            downloads[3].download_button(
                "Download case report (TXT)",
                data=case_report.encode("utf-8"),
                file_name=f"patellavision_surgical_case_report_slice_{slice_index + 1:04d}.txt",
                mime="text/plain",
            )
            downloads[4].download_button(
                "Download research review log (CSV)",
                data=make_review_log_csv(review_rows),
                file_name="patellavision_research_review_log.csv",
                mime="text/csv",
            )

            st.subheader("Case summary for discussion")
            with st.container():
                st.markdown(f"**Slice:** {slice_index + 1} of {len(ordered_slices)}")
                st.markdown(f"**Area highlighted:** {mask_ratio:.1f}% of image")
                st.markdown(f"**Approximate local focus:** {bbox_width} x {bbox_height} pixels")
                st.markdown(f"**Review status:** {review_records.get(slice_index, {}).get('status', status_options[0])}")
                st.markdown(f"**Surgeon summary:** {case_context.get('surgeon_summary', 'No surgeon notes recorded.')}")
                st.markdown(f"**Patient note:** {case_context.get('patient_note') or editable_patient_note}")

show_test_metrics()

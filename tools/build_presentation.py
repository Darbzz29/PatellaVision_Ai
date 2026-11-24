"""Build the editable PatellaVision AI project presentation."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PatellaVision_AI_Presentation.pptx"

NAVY = RGBColor(17, 34, 62)
BLUE = RGBColor(49, 112, 190)
TEAL = RGBColor(16, 151, 155)
INK = RGBColor(35, 48, 66)
MUTED = RGBColor(93, 108, 126)
PALE = RGBColor(241, 246, 251)
WHITE = RGBColor(255, 255, 255)
GOLD = RGBColor(241, 183, 72)

WIDTH = 13.333
HEIGHT = 7.5


def add_box(slide, x, y, width, height, color, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(
        shape_type,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    if radius:
        shape.adjustments[0] = 0.08
    return shape


def add_text(
    slide,
    text,
    x,
    y,
    width,
    height,
    *,
    size=18,
    color=INK,
    bold=False,
    font="Aptos",
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0.04,
):
    shape = slide.shapes.add_textbox(
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def add_bullets(slide, items, x, y, width, *, line_height=0.58, size=17):
    for index, item in enumerate(items):
        row_y = y + index * line_height
        add_box(slide, x, row_y + 0.12, 0.1, 0.1, TEAL, radius=True)
        add_text(slide, item, x + 0.22, row_y, width - 0.22, line_height - 0.04, size=size)


def add_image_contain(slide, path: Path, x, y, width, height):
    if not path.is_file():
        return add_text(
            slide,
            f"Figure not found:\n{path.relative_to(ROOT)}",
            x,
            y,
            width,
            height,
            size=12,
            color=MUTED,
        )
    from PIL import Image

    with Image.open(path) as image:
        image_width, image_height = image.size
    scale = min(width / image_width, height / image_height)
    draw_width = image_width * scale
    draw_height = image_height * scale
    left = x + (width - draw_width) / 2
    top = y + (height - draw_height) / 2
    return slide.shapes.add_picture(
        str(path),
        Inches(left),
        Inches(top),
        width=Inches(draw_width),
        height=Inches(draw_height),
    )


def new_slide(prs: Presentation, title: str, section: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    add_box(slide, 0, 0, WIDTH, 0.18, BLUE)
    add_text(slide, section.upper(), 0.55, 0.36, 5, 0.28, size=10, color=TEAL, bold=True)
    add_text(slide, title, 0.55, 0.7, 12.2, 0.55, size=27, color=NAVY, bold=True)
    add_box(slide, 0.55, 1.35, 0.7, 0.06, TEAL)
    slide_number = len(prs.slides)
    add_text(
        slide,
        f"PATELLAVISION AI  |  RESEARCH PROTOTYPE  |  {slide_number:02d}",
        0.55,
        7.15,
        12.2,
        0.18,
        size=8,
        color=MUTED,
    )
    return slide


def add_metric(slide, x, y, label, value, note=""):
    add_box(slide, x, y, 2.45, 1.28, PALE, radius=True)
    add_text(slide, label.upper(), x + 0.16, y + 0.12, 2.1, 0.2, size=9, color=MUTED, bold=True)
    add_text(slide, value, x + 0.16, y + 0.34, 2.1, 0.46, size=25, color=NAVY, bold=True)
    if note:
        add_text(slide, note, x + 0.16, y + 0.88, 2.1, 0.25, size=9, color=MUTED)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(WIDTH)
    prs.slide_height = Inches(HEIGHT)
    prs.core_properties.title = (
        "PatellaVision AI: Reliability Characterization of a U-Net Baseline"
    )
    prs.core_properties.subject = (
        "Single-volume exploratory knee MRI segmentation and perturbation study"
    )
    prs.core_properties.author = "PatellaVision AI project team"

    # 1. Title
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    add_box(slide, 0.8, 0.85, 0.13, 4.65, TEAL)
    add_text(slide, "PATELLAVISION AI", 1.2, 1.05, 10.8, 0.35, size=14, color=GOLD, bold=True)
    add_text(
        slide,
        "Reliability characterization of a U-Net baseline for binary knee MRI bone segmentation",
        1.2,
        1.6,
        10.8,
        1.5,
        size=32,
        color=WHITE,
        bold=True,
    )
    add_text(
        slide,
        "A single-volume exploratory study under controlled image-quality perturbations",
        1.2,
        3.35,
        10.5,
        0.55,
        size=19,
        color=RGBColor(197, 213, 230),
    )
    add_text(
        slide,
        "Team: [Member 1 name]  |  [Member 2 name]\nCourse / institution: [Add details]",
        1.2,
        4.5,
        9.5,
        0.65,
        size=15,
        color=WHITE,
    )
    add_text(
        slide,
        "Experimental segmentation only — not a diagnostic or clinical decision tool.",
        1.2,
        6.55,
        10.8,
        0.35,
        size=13,
        color=GOLD,
        bold=True,
    )

    # 2. Problem and contribution
    slide = new_slide(prs, "Problem, research question, and contribution", "01 / Study focus")
    add_box(slide, 0.65, 1.72, 5.85, 4.9, PALE, radius=True)
    add_text(slide, "Research question", 0.95, 1.98, 5.2, 0.35, size=20, color=NAVY, bold=True)
    add_text(
        slide,
        "How do this trained U-Net baseline's Dice and IoU change when held-out MRI slices receive controlled synthetic noise, contrast reduction, or blur?",
        0.95,
        2.45,
        5.1,
        1.05,
        size=17,
    )
    add_text(slide, "Project-specific contribution", 0.95, 3.75, 5.2, 0.35, size=20, color=NAVY, bold=True)
    add_bullets(
        slide,
        [
            "One documented binary U-Net baseline.",
            "Clean-versus-perturbed comparison on the same held-out block.",
            "Per-slice scores and visual failure examples.",
        ],
        0.98,
        4.25,
        5.1,
        line_height=0.55,
        size=15,
    )
    add_box(slide, 6.85, 1.72, 5.8, 4.9, NAVY, radius=True)
    add_text(slide, "Prior work and novelty boundary", 7.2, 2.05, 5.1, 0.38, size=19, color=WHITE, bold=True)
    add_text(
        slide,
        "Prior work establishes this area: U-Net [1], knee MRI segmentation reviews [4], and sequence-dependent image properties [6]. MRNet [8] studied report-derived findings such as ACL and meniscal tears, not whether surgery is needed.",
        7.2,
        2.55,
        5.0,
        1.75,
        size=15,
        color=WHITE,
    )
    add_text(
        slide,
        "Our contribution: a controlled sensitivity study of one binary U-Net baseline. No new architecture or clinical decision claim.",
        7.2,
        4.65,
        5.0,
        1.1,
        size=15,
        color=GOLD,
        bold=True,
    )
    add_text(slide, "Full references are listed in REPORT.md.", 7.2, 6.05, 5.0, 0.25, size=10, color=RGBColor(197, 213, 230))

    # 3. Dataset
    slide = new_slide(prs, "Dataset and pairing verification", "02 / Data")
    add_bullets(
        slide,
        [
            "286 DICOM slices and 286 PNG masks; 512 x 512 pixels.",
            "Numeric DICOM suffix _0–_285 paired with Mask_001–Mask_286.",
            "Dimensions checked; sampled overlays visually reviewed.",
            "Dataset README describes one 286-slice volume.",
        ],
        0.75,
        1.85,
        5.1,
        line_height=0.78,
        size=16,
    )
    add_box(slide, 6.05, 1.72, 6.55, 4.95, PALE, radius=True)
    add_image_contain(slide, ROOT / "results" / "day2_pairing_preview.png", 6.2, 1.85, 6.25, 4.55)
    add_text(slide, "Sampled MRI / annotation overlays (red)", 6.25, 6.35, 6.2, 0.25, size=11, color=MUTED, align=PP_ALIGN.CENTER)

    # 4. Preprocessing and split
    slide = new_slide(prs, "Preprocessing and ordered split", "03 / Method")
    add_box(slide, 0.72, 1.78, 5.8, 4.95, PALE, radius=True)
    add_text(slide, "Image preparation", 1.0, 2.05, 5.1, 0.35, size=20, color=NAVY, bold=True)
    add_bullets(
        slide,
        [
            "Decode DICOM pixel data; invert MONOCHROME1 when present.",
            "Clip each slice to its 1st–99th intensity percentiles; scale to [0, 1].",
            "Resize MRI to 128 x 128 (bilinear); masks to 128 x 128 (nearest).",
        ],
        1.0,
        2.62,
        5.0,
        line_height=0.92,
        size=15,
    )
    add_box(slide, 6.82, 1.78, 5.75, 4.95, PALE, radius=True)
    add_text(slide, "Contiguous slice blocks", 7.12, 2.05, 5.0, 0.35, size=20, color=NAVY, bold=True)
    split_rows = [
        ("TRAIN", "197 slices", "0–196", BLUE),
        ("GAP", "2 omitted", "197–198", MUTED),
        ("VALIDATION", "42 slices", "199–240", TEAL),
        ("GAP", "2 omitted", "241–242", MUTED),
        ("TEST", "43 slices", "243–285", GOLD),
    ]
    for i, (name, count, indices, color) in enumerate(split_rows):
        y = 2.65 + i * 0.66
        add_box(slide, 7.15, y + 0.04, 0.16, 0.42, color, radius=True)
        add_text(slide, name, 7.48, y, 1.65, 0.45, size=13, color=NAVY, bold=True)
        add_text(slide, count, 9.1, y, 1.45, 0.45, size=14)
        add_text(slide, indices, 10.6, y, 1.45, 0.45, size=13, color=MUTED)
    add_text(slide, "Slices remain correlated even with contiguous blocks and gaps.", 7.15, 6.15, 4.95, 0.35, size=11, color=MUTED)

    # 5. Model and training
    slide = new_slide(prs, "U-Net baseline and training", "04 / Model")
    add_box(slide, 0.72, 1.8, 5.55, 4.9, NAVY, radius=True)
    add_text(slide, "Encoder", 1.05, 2.12, 1.3, 0.3, size=15, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "Context\nfeatures", 1.05, 3.05, 1.3, 0.8, size=15, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, "Bottleneck", 2.75, 3.0, 1.4, 0.6, size=15, color=GOLD, bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    add_text(slide, "Decoder", 4.55, 2.12, 1.25, 0.3, size=15, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    for x in [1.32, 2.08, 2.85, 3.62, 4.39]:
        add_box(slide, x, 4.65, 0.5, 0.48, TEAL if x != 2.85 else BLUE, radius=True)
    add_text(slide, "MRI", 1.2, 5.5, 0.75, 0.3, size=13, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, "Binary mask", 4.2, 5.5, 1.7, 0.3, size=13, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, "Skip connections preserve spatial detail", 1.1, 6.1, 4.8, 0.28, size=12, color=RGBColor(197, 213, 230), align=PP_ALIGN.CENTER)
    add_box(slide, 6.65, 1.8, 5.9, 4.9, PALE, radius=True)
    add_text(slide, "Training configuration", 6.98, 2.12, 5.0, 0.35, size=20, color=NAVY, bold=True)
    add_bullets(
        slide,
        [
            "PyTorch CPU; 20 epochs; batch size 8.",
            "Adam, learning rate 0.001; base channels 8.",
            "BCE-with-logits + soft Dice loss.",
            "Best validation Dice checkpoint: epoch 10 (0.9286).",
            "The 43-slice test block was not used for selection.",
        ],
        7.0,
        2.7,
        5.0,
        line_height=0.63,
        size=15,
    )
    add_text(slide, "Baseline architecture follows established U-Net work [1].", 7.0, 6.15, 5.0, 0.28, size=11, color=MUTED)

    # 6. Learning curves
    slide = new_slide(prs, "Training history", "05 / Optimization")
    add_image_contain(slide, ROOT / "results" / "graphs" / "training_curves.png", 0.9, 1.62, 8.5, 4.9)
    add_box(slide, 9.7, 1.82, 2.75, 4.5, PALE, radius=True)
    add_text(slide, "Observed", 10.0, 2.12, 2.1, 0.35, size=19, color=NAVY, bold=True)
    add_bullets(
        slide,
        [
            "Training Dice rose over epochs.",
            "Validation Dice fluctuated.",
            "Checkpoint chosen at epoch 10.",
            "One split and one seed limit conclusions.",
        ],
        10.0,
        2.75,
        2.15,
        line_height=0.74,
        size=13,
    )
    add_text(slide, "Training and validation history from the recorded 20-epoch CPU run.", 1.0, 6.6, 8.4, 0.25, size=11, color=MUTED, align=PP_ALIGN.CENTER)

    # 7. Test evaluation
    slide = new_slide(prs, "Held-out test evaluation", "06 / Results")
    add_metric(slide, 0.8, 1.82, "Mean slice Dice", "0.9622", "43 test slices")
    add_metric(slide, 3.52, 1.82, "Mean slice IoU", "0.9276", "43 test slices")
    add_metric(slide, 0.8, 3.35, "Pooled Dice", "0.9633", "pixel-aggregated")
    add_metric(slide, 3.52, 3.35, "Pooled IoU", "0.9292", "pixel-aggregated")
    add_box(slide, 6.45, 1.72, 6.15, 4.98, PALE, radius=True)
    add_image_contain(slide, ROOT / "results" / "predictions" / "test_slice_285.png", 6.6, 1.92, 5.85, 4.35)
    add_text(slide, "Slice 285 was the lowest-scoring test slice (Dice 0.8965).", 6.65, 6.35, 5.7, 0.25, size=11, color=MUTED, align=PP_ALIGN.CENTER)
    add_text(slide, "Exploratory slice-level holdout from one volume; not a patient-level test.", 0.85, 5.55, 5.1, 0.55, size=14, color=NAVY, bold=True)

    # 8. Reliability results
    slide = new_slide(prs, "Reliability under synthetic image changes", "07 / Controlled experiment")
    add_image_contain(slide, ROOT / "results" / "reliability" / "reliability_comparison.png", 0.7, 1.65, 8.6, 4.9)
    add_box(slide, 9.55, 1.82, 3.0, 4.65, NAVY, radius=True)
    add_text(slide, "Mean Dice", 9.9, 2.15, 2.2, 0.35, size=18, color=WHITE, bold=True)
    add_text(slide, "Clean  0.9622\nNoise 0.10  0.9410\nContrast 0.75  0.8367\nContrast 0.50  0.1869\nBlur 1 px  0.8780\nBlur 2 px  0.6222", 9.9, 2.72, 2.3, 2.55, size=14, color=WHITE)
    add_text(slide, "Synthetic normalized-image changes; not calibrated scanner simulations.", 9.9, 5.55, 2.25, 0.7, size=11, color=GOLD)
    add_text(slide, "Same checkpoint and ground truth; 43 held-out slices per condition.", 0.8, 6.65, 8.5, 0.2, size=11, color=MUTED, align=PP_ALIGN.CENTER)

    # 9. Failure-case visualization
    slide = new_slide(prs, "Where performance degrades", "08 / Qualitative analysis")
    add_image_contain(slide, ROOT / "results" / "reliability" / "reliability_worst_slice.png", 0.85, 1.55, 7.4, 5.45)
    add_box(slide, 8.55, 1.85, 3.95, 4.8, PALE, radius=True)
    add_text(slide, "Example: test slice 285", 8.9, 2.18, 3.2, 0.4, size=19, color=NAVY, bold=True)
    add_bullets(
        slide,
        [
            "Clean Dice: 0.897.",
            "Mild noise changes little in this example.",
            "Strong contrast loss and blur create larger errors.",
            "Visual example is illustrative; aggregate table reports all 43 slices.",
        ],
        8.9,
        2.9,
        3.15,
        line_height=0.78,
        size=14,
    )

    # 10. Application and limitations
    slide = new_slide(prs, "Demo workflow and limits", "09 / Responsible use")
    add_box(slide, 0.72, 1.78, 5.75, 4.95, PALE, radius=True)
    add_text(slide, "Streamlit demonstration", 1.02, 2.08, 5.0, 0.4, size=21, color=NAVY, bold=True)
    add_bullets(
        slide,
        [
            "Upload one MR series; validate and order its slices.",
            "Inspect a selected MRI slice, mask, and adjustable overlay.",
            "View scan metadata and pixel-based mask statistics.",
            "Save notes/checklist; export images and case summaries.",
            "Run: streamlit run app\\app.py",
        ],
        1.03,
        2.8,
        4.95,
        line_height=0.68,
        size=13,
    )
    add_box(slide, 6.78, 1.78, 5.75, 4.95, NAVY, radius=True)
    add_text(slide, "Limitations", 7.12, 2.08, 4.9, 0.4, size=21, color=WHITE, bold=True)
    add_bullets(
        slide,
        [
            "One volume; no patient-level external validation.",
            "Merged foreground; no individual bone or condition labels.",
            "Pixel statistics are not anatomical or surgical measurements.",
            "Research prototype; no diagnosis or surgery recommendation.",
        ],
        7.13,
        2.8,
        4.9,
        line_height=0.78,
        size=15,
    )

    # 11. Team work and next steps
    slide = new_slide(prs, "Both team workstreams completed", "10 / Team and conclusion")
    add_box(slide, 0.72, 1.78, 5.75, 3.7, PALE, radius=True)
    add_text(slide, "Member 1 — AI / ML", 1.04, 2.08, 5.0, 0.42, size=21, color=NAVY, bold=True)
    add_bullets(
        slide,
        [
            "Verified numeric MRI-mask pairing and ordered split.",
            "Implemented preprocessing, U-Net, and CPU training.",
            "Selected checkpoint and measured held-out Dice / IoU.",
        ],
        1.05,
        2.78,
        4.9,
        line_height=0.72,
        size=15,
    )
    add_box(slide, 6.78, 1.78, 5.75, 3.7, PALE, radius=True)
    add_text(slide, "Member 2 — Image processing / research", 7.1, 2.08, 5.0, 0.65, size=19, color=NAVY, bold=True)
    add_bullets(
        slide,
        [
            "Ran noise, contrast, and blur experiments.",
            "Prepared overlays, figures, and Streamlit demo.",
            "Prepared prior-work context, report, and slides.",
        ],
        7.11,
        2.92,
        4.9,
        line_height=0.72,
        size=15,
    )
    add_box(slide, 0.72, 5.75, 11.82, 0.95, NAVY, radius=True)
    add_text(
        slide,
        "Conclusion: a working baseline and a focused exploratory sensitivity study. Next: multiple licensed volumes, patient-level splits, repeated runs, and external evaluation.",
        1.0,
        5.98,
        11.25,
        0.52,
        size=15,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_text(
        slide,
        "Future surgery-decision research needs a separately approved dataset with clinical context and clinician-reviewed treatment/outcome labels. Add actual names and confirm individual contributions.",
        1.0,
        6.78,
        11.2,
        0.3,
        size=9,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )

    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())

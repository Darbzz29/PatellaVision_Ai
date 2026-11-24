# PatellaVision AI

**Reliability-aware knee MRI image processing and binary bone segmentation**

This is an experimental computer-vision research prototype. It segments the
foreground annotated in 2D knee MRI slices. It is not a medical device and does
not diagnose a condition or predict a clinical outcome.

## What is implemented

- DICOM slice and PNG mask pairing by numeric slice order.
- Per-slice intensity normalization and binary-mask preprocessing.
- A compact PyTorch U-Net.
- Contiguous train/validation/test blocks with two omitted slices between blocks.
- CPU training, validation-based checkpoint selection, and a held-out test
  evaluation.
- Controlled Gaussian noise, contrast, and blur experiments.
- A Streamlit MR-series review interface with slice navigation, side-by-side,
  overlay, and mask views; guided review instructions; non-identifying scan
  metadata; pixel-based mask statistics; per-slice research flags and notes;
  a patient explanation template; a clinician-entered case-discussion
  checklist and notes; and PNG/TXT/CSV exports for research.



These are role-based responsibilities, not verified individual authorship.
Add team member names and confirm actual contributions before submission.

## Project-specific contribution

This project does not introduce a new U-Net architecture. Its focused,
exploratory contribution is a clean-versus-perturbed sensitivity comparison
for one trained binary-segmentation baseline, including per-slice metrics and
visual examples for six synthetic noise, contrast, and blur conditions.
Because the available dataset is one volume, this does not establish
patient-level generalization or global research novelty. See
[REPORT.md](./REPORT.md) for the scope, prior work, and limitations.

## Project layout

```text
app/app.py                 Streamlit demo
dataset/<dataset>/         Original DICOMs, masks, README, and license
models/best_unet.pt        Best validation-Dice checkpoint
results/                   Metrics, figures, prediction overlays, and history
src/data.py                Pairing, preprocessing, split, dataset
src/model.py               U-Net
src/train.py               Training entry point
src/evaluate.py            Held-out test evaluation
src/reliability.py         Controlled robustness experiments
tools/build_presentation.py Editable PowerPoint deck generator
PatellaVision_AI_Presentation.pptx  11-slide editable presentation
```

## Environment setup (Windows PowerShell)

Open PowerShell in this project folder:

```powershell
cd "C:\Users\Deborah\Desktop\PatellaVision_AI\PatellaVision_AI"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The experiments recorded here ran on CPU with Python 3.13.3 and PyTorch
2.15.0+cpu. No NVIDIA GPU was used.

## Run the project

Train the U-Net. The defaults reproduce the recorded 20-epoch CPU training
configuration:

```powershell
.\.venv\Scripts\python.exe -m src.train
```

Evaluate the saved best checkpoint once on the held-out test block:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```

Run the controlled image-quality experiments:

```powershell
.\.venv\Scripts\python.exe -m src.reliability
```

Start the research review demo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

Then open the local address printed by Streamlit, usually
`http://localhost:8501`.

Upload all slices from one MR DICOM series to browse them in spatial order;
the model still predicts one 2D slice at a time. Reviewers can flag a slice
for manual correction, add research notes, and download a review log. The
binary output merges all annotated foreground and cannot identify individual
bones or calculate surgical measurements. PNG and CSV exports are for
research review only; the app does not generate a clinical report or DICOM-SEG
object. The checkpoint was evaluated on slices from one volume only. Do not
use the prototype or its outputs for diagnosis, treatment, or surgical
planning. Use only de-identified data or data in an environment approved by
your organization.

Rebuild the editable presentation after changing the report figures or results:

```powershell
.\.venv\Scripts\python.exe -m tools.build_presentation
```

## Recorded results

The best checkpoint was selected at epoch 10 by validation Dice. On the
43-slice contiguous test block, the recorded mean per-slice Dice was 0.9622 and
mean per-slice IoU was 0.9276. These are exploratory results from a single
volume, not evidence of performance on unseen patients.

The clean test set was also used for controlled perturbation experiments:

| Input condition | Mean Dice | Mean IoU |
|---|---:|---:|
| Unmodified | 0.9622 | 0.9276 |
| Gaussian noise, sigma 0.05 | 0.9575 | 0.9190 |
| Gaussian noise, sigma 0.10 | 0.9410 | 0.8894 |
| Contrast factor 0.75 | 0.8367 | 0.7225 |
| Contrast factor 0.50 | 0.1869 | 0.1032 |
| Gaussian blur, sigma 1 px | 0.8780 | 0.7839 |
| Gaussian blur, sigma 2 px | 0.6222 | 0.4536 |

These are synthetic alterations of normalized images, not scanner-domain
validation.

## Limitations and responsible use

- The supplied dataset contains 286 slices from one volume. A contiguous
  slice-level holdout cannot establish generalization to different patients,
  scanners, protocols, or institutions.
- Neighboring slices are anatomically correlated even when the split is
  contiguous and separated by small gaps.
- At model input resolution, all nonzero mask pixels are treated as one binary
  foreground class; the model does not distinguish individual bone classes.
- The reliability alterations are controlled synthetic changes, not a
  clinically calibrated image-quality study.
- The app browses one MR DICOM series and displays a prediction for the
  selected 2D slice. It is intended for teaching and research review only, not
  diagnosis, treatment, or surgical planning.
- The dataset folder includes a `LICENSE` file beginning with GPL-3.0 text.
  Confirm the dataset provider's terms and the scope of that license before
  redistributing images, masks, or derived data. This project does not include
  the dataset license as a claim about ownership of the medical data.



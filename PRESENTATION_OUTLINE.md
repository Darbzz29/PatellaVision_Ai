# PatellaVision AI — 11-slide presentation outline

Use the supplied figures in `results/`. State that all measured scores are
from one volume and are exploratory. Do not describe the demo as a diagnostic
tool.

## Core contribution statement

> “We do not propose a new U-Net. Our project-specific contribution is an
> empirical sensitivity characterization of one trained binary-segmentation
> baseline: we compare the same checkpoint on clean and synthetically
> perturbed held-out slices, with per-slice scores and visual failure cases.
> This is exploratory evidence from one volume, not a claim of global novelty
> or patient-level generalization.”

## Slide 1 — Title and scope

- PatellaVision AI: reliability characterization of a U-Net baseline for
  binary knee MRI segmentation.
- Subtitle: a single-volume exploratory study under controlled
  image-quality perturbations.
- Team member names and course/institution.
- Scope: segment annotated foreground; no diagnosis or prediction.
- **Say:** “This is an experimental segmentation prototype, not a clinical
  decision system.”

## Slide 2 — Problem, question, and project contribution

- Input: one 2D knee MRI slice.
- Output: a binary mask of the annotated region.
- **Research question:** how do the same model's Dice/IoU scores change under
  specified synthetic noise, contrast reduction, and blur?
- Contribution: one documented baseline, clean-versus-perturbed comparison,
  per-slice metrics, and visual failure cases on one held-out slice block.
- Prior work includes MRI-based knee segmentation reviews and sequence
  comparisons. MRNet studied report-derived findings such as ACL and meniscal
  tears, not whether surgery is needed. See REPORT.md for citations.
- The U-Net architecture is established prior work; do not claim a new model,
  first-of-its-kind result, or global novelty.
- **Visual:** a clean MRI, ground truth, and predicted mask from the held-out
  montage.

## Slide 3 — Dataset and pairing checks

- KneeBones3Dify-Annotated-Dataset; README describes 286 slices at 512 x 512.
- 286 DICOM files and 286 PNG masks.
- DICOM suffix `_0` through `_285` paired with `Mask_001.png` through
  `Mask_286.png` by numeric slice order.
- Dimensions checked; sampled overlays reviewed.
- **Caveat:** the available data is one volume, not a patient cohort.
- **Visual:** `results/day2_pairing_preview.png`.

## Slide 4 — Preprocessing and split

- DICOM decode; MONOCHROME1 inversion when applicable.
- Per-slice 1st/99th percentile clipping and scaling to [0, 1].
- Images bilinear-resized; binary masks nearest-neighbor-resized to 128 x 128.
- Contiguous blocks: train 197, validation 42, test 43; two omitted slices at
  each boundary.
- **Caveat:** contiguous slices remain correlated.

## Slide 5 — U-Net architecture

- Four-level encoder-decoder with skip connections.
- Two convolution / batch-normalization / ReLU operations per block.
- One MRI input channel; one binary-foreground logit output.
- BCE-with-logits plus soft Dice training loss.
- **Reference:** Ronneberger et al., MICCAI 2015.

## Slide 6 — Training setup

- CPU-only PyTorch; 20 epochs, batch size 8, Adam at 0.001.
- Input size 128 x 128; base channels 8.
- Best checkpoint selected by validation Dice at epoch 10.
- Test block not used for training or model selection.
- **Visual:** `results/graphs/training_curves.png`.

## Slide 7 — Held-out test evaluation

- 43 test slices, indices 243-285.
- Mean per-slice Dice 0.9622; mean per-slice IoU 0.9276.
- Pooled pixel Dice 0.9633; pooled pixel IoU 0.9292.
- Lowest per-slice Dice 0.8965 at slice 285.
- **Visual:** `results/test_predictions_montage.png`.
- **Caveat:** these results do not measure performance on unseen patients.

## Slide 8 — Reliability experiment

- Fixed model and masks; controlled synthetic input alterations.
- Noise sigma 0.05/0.10; contrast factor 0.75/0.50; blur sigma 1/2 px.
- Mean Dice: clean 0.9622; noise 0.05 0.9575; noise 0.10 0.9410;
  contrast 0.75 0.8367; contrast 0.50 0.1869; blur 1 px 0.8780;
  blur 2 px 0.6222.
- **Visual:** `results/reliability/reliability_comparison.png`.
- **Caveat:** synthetic changes are not calibrated scanner simulations.

## Slide 9 — Demo, limitations, and responsible use

- Streamlit accepts one MR DICOM series, validates and sorts slices, and
  supports slice navigation, side-by-side, adjustable overlay, mask-only
  review, guided review steps, scan metadata, pixel-based mask statistics,
  per-slice research notes, a patient explanation template, a case-discussion
  checklist, and PNG/TXT/CSV exports.
- The model still predicts one 2D binary foreground mask at a time; it does
  not distinguish bones, diagnose a condition, determine whether surgery is
  needed, or provide surgical measurements.
- One volume; no patient-separated external validation.
- Binary union mask; no separate bone labels.
- No clinical evaluation, diagnosis, or treatment recommendation.
- **Live demo:** run `python -m streamlit run app\app.py`.

## Slide 10 — Conclusion and next research steps

- Working end-to-end loader, model, evaluation, reliability experiments, and
  demonstration.
- Project-specific contribution: a controlled empirical sensitivity
  characterization of this baseline, not a novel architecture.
- Report the measured scores as exploratory results only.
- Next: obtain properly licensed multi-patient data, split by patient, verify
  annotation semantics, run repeated experiments, and evaluate externally.
- MRNet-style report-derived finding labels could support a separate research
  classification task, but they are not surgical-indication labels. A
  surgery-decision study would need an appropriately approved cohort with
  clinical context and clinician-reviewed treatment/outcome labels.
- Thank the audience; invite questions.

## Slide 11 — Team work completed

- **Member 1 - AI/ML:** verified numeric slice pairing; built preprocessing,
  contiguous split, U-Net and CPU training; evaluated held-out Dice/IoU.
- **Member 2 - image processing, app, and research:** ran controlled
  perturbations; prepared visualizations; built Streamlit demo; drafted
  literature context, report, and presentation.
- Add each member's name only after the team confirms actual contributions.
- Both members should be ready to explain the full pipeline and limitations.

## Likely viva questions

1. **Why U-Net?** Its encoder-decoder structure with skip connections is a
   common biomedical segmentation baseline [1].
2. **Why a binary mask?** The supplied PNGs were treated as foreground versus
   background; individual bone classes are not distinguished in this model.
3. **Why use contiguous splits?** To avoid randomly interleaving nearby slices,
   while acknowledging that one-volume slices are still correlated.
4. **Why report Dice and IoU?** They quantify overlap, with Dice emphasizing
   shared foreground relative to the two mask sizes and IoU relative to their
   union.
5. **What does the test score mean?** Overlap on this held-out slice block from
   the same volume only; it is not patient-level generalization.
6. **Which perturbation had the largest effect?** Contrast factor 0.50, where
   mean Dice fell to 0.1869 in this synthetic experiment.
7. **Is the app clinically usable?** No. It is a research demonstration only.
8. **What would improve the study most?** Multiple properly licensed patients
   and a patient-level held-out test set.
9. **What is unique about this project?** Its project-specific contribution is
   the measured clean-versus-synthetic-perturbation sensitivity comparison
   with per-slice evidence for this model and dataset split. It does not claim
   that U-Net, knee-bone segmentation, or robustness testing is new in the
   wider literature.
10. **Can this model say if surgery is needed?** No. It predicts one merged
    binary mask and has no surgical-indication labels or clinical validation.

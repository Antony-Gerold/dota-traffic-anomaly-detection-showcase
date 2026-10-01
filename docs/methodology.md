# Methodology

Everything here follows the final report ([PDF](../report/DoTA_Project4_FinalReport.pdf)).

## 1. Problem and hypothesis

The DoTA dataset [1] contains 4,677 dashcam videos annotated with anomaly start and end times, bounding-box tracks of the anomalous object, and one of nine accident categories per video. The original DoTA method (FOL-Ensemble) is unsupervised and works at the frame level: it predicts when an anomaly happens but does not explicitly identify which object is anomalous. The paper's STAUC metric penalizes methods that get the timing right but fail to localize the anomalous object.

The hypothesis was that a detection-first, object-level pipeline with supervision targeting STAUC would localize better than the frame-level baseline.

## 2. Training and evaluation protocol

- 5-fold GroupKFold over all 4,677 videos, grouped by video id, so frames from one video never appear in both training and test folds. All in-distribution per-detection AUC, frame AUC, and STAUC numbers are pooled out-of-fold predictions.
- The same split is reused at every stage. Classifier A trains five LightGBM models (one per fold, each on about 3,742 videos, scoring the held-out 935). Stage-2 trains on Classifier A's out-of-fold scores. Classifier B uses per-track features from training-fold anomaly windows and is evaluated on held-out folds.
- For CCD and Nexar, a single Classifier A was trained on all 4,677 DoTA videos and applied without retraining or fine-tuning.
- Yao et al. report on DoTA's official 1,402-video validation split. Both protocols evaluate on unseen videos; the 5-fold approach evaluates on more videos with proportionally less training data per fold.

## 3. Detection and tracking

**YOLOv8m at 1280×720** [2]. Six COCO classes: person, bicycle, car, motorcycle, bus, truck. Confidence threshold 0.25, IoU threshold for tracking 0.45. YOLOv8n misses too many small objects (a vehicle 30 m ahead can be under 50 pixels); YOLOv8l and YOLOv8x give marginal gains at three to six times the inference cost. Single-frame inference is roughly 50 ms on a Tesla T4.

**ByteTrack** [3] with default Ultralytics settings. Its second association stage matches low-confidence detections that most trackers discard, which matters because occlusion, motion blur, and unusual lighting near a collision are exactly when confidence drops. SORT [11] or DeepSORT would lose these tracks.

## 4. Features

A 57-dimensional feature vector was designed per detection, in five groups:

| Group | Features | Content |
|---|---|---|
| Trajectory | 18 | Normalized centre and bottom position, box area and aspect ratio, velocity, speed, direction, acceleration, jerk, track age, track stability, YOLO class |
| Interaction | 9 | Min and mean nearest-neighbour distance, max and mean closing rate, max IoU with another object, max relative speed, count of others in frame, TTC-derived features |
| Time-to-collision | 2 | Inverse TTC from the rate of bounding-box growth (looming model) |
| Temporal rolling | 24 | Rolling max (5 and 10 frames), mean (10), and std (10) of six base features, plus cumulative direction change and 5-frame speed delta |
| Residual | 4 | Actual minus predicted next-frame cx, cy, vx, vy |

Four optical-flow-decoupled features were left at zero for compute reasons, so the final vector is **53 numbers per detection**. With 2,744,200 detections this gives a 2.74M × 53 training matrix.

## 5. Classifier A: per-detection anomaly scoring

LightGBM [4], chosen because the features are tabular, the training set is 2.74 million samples, and scoring all detections takes about 30 seconds on CPU. Configuration: 300 boosting rounds, learning rate 0.05, max depth 6, 31 leaves, min child samples 50, scale_pos_weight set to the negative-to-positive ratio.

**Label noise.** The first labelling rule marked every detection in an anomaly window as positive and gave AUC 0.88. A typical anomaly frame has five to fifteen detected objects but only one ground-truth anomalous object, so most "positives" were bystanders. For each detection in the anomaly window, IoU with the ground-truth anomalous box was computed and detections below 0.3 were relabelled negative. This relabelled **692,412 detections** and raised AUC from **0.88 to 0.9151**. The 0.3 threshold was chosen empirically: 0.5 was too strict under heavy occlusion, 0.1 admitted nearby bystanders.

## 6. Stage-2 frame aggregation

Taking the max per-detection score per frame gave frame AUC 0.65. With a per-detection false-positive rate of about 5% at threshold 0.5, a 10-detection normal frame has roughly a 40% chance that at least one detection exceeds 0.5, which inflates the noise floor.

Stage-2 is a second classifier on a 22-dimensional per-frame vector: max, mean, 90th and 75th percentile of detection scores; counts above 0.3, 0.5, and 0.7; the same statistics over 5-frame and 10-frame trailing windows; and an exponential moving average of the per-frame max (smoothing factor 0.3). A real anomaly produces a sustained multi-detection pattern; a false positive is a single-frame spike. Stage-2 raised frame AUC from 0.65 to **0.7272**.

## 7. Classifier B: anomaly category

Eight reliably labelled categories: turning, lateral, oncoming, moving-ahead-or-waiting, pedestrian, leave-to-right, leave-to-left, start-stop-or-stationary. Unknown and obstacle were excluded because those labels are heterogeneous.

- Trajectory features alone on a 200-video pilot: top-1 0.27, top-3 0.71.
- Hierarchical ego vs non-ego split: failed, since the ego decision itself reached only AUC 0.63. Balanced undersampling and one-vs-rest classifiers did not help enough.
- **Motion signature** per track over the whole anomaly window: Fourier descriptors (4 complex coefficients), an 8-bin direction histogram, speed-profile statistics, and the per-track Classifier A score curve resampled to 10 positions plus argmax position and peak score. With a two-stage A3 classifier (turning vs rest, then fine-grained), top-1 reached 0.46 on full DoTA.
- **DINOv2 fusion** [7]. For each anomalous track, 5 frames sampled across the window, two crops per frame (tight object crop and a 1.5× context crop, resized to 518×518). DINOv2-small gives 384-d per crop, mean-pooled over the 10 crops and PCA-reduced to 128-d (84.2% variance retained). Motion (64-d) and DINOv2 (128-d) are concatenated to 192-d and fed to the same A3 classifier: **top-1 0.505, top-3 0.802**.

## 8. Cross-dataset evaluation

Same DoTA-trained Classifier A and Stage-2, no retraining.

- **Nexar** [6]: 1,500 dashcam videos with a video-level collision label and time of event. Frames sampled at stride 6; frames within ±2 s of the event are positive in collision videos.
- **CCD** [5]: 1,500 crash videos, each 50 frames at 10 fps, with per-frame binary crash labels.

## 9. Visual-only and VLM methods tried before the pivot

On a 200-video DoTA subset:

- CLIP text-anchor matching (ViT-B/32): frame AUC 0.557.
- DINOv2 perceptual-curvature scoring: frame AUC 0.637.
- Four-method unsupervised fusion: fAUC 0.657, STAUC 0.414, AUC-STAUC gap 0.243.
- Qwen2-VL-2B zero-shot: 0.520 single-frame, 0.513 three-frame, 0.510 per-object crop (6.9% parse failures).

Frame-level methods broadcast one score across all objects instead of localizing the anomalous one, which is what STAUC penalizes. Switching to the object-level pipeline closed the gap to 0.231 and lifted STAUC from 0.414 to 0.4966.

Bracketed numbers refer to [references.md](references.md).

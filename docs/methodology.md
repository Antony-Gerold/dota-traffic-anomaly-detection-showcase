# Methodology

## 1. Problem framing

**Input:** dashcam video. **Output:** per-frame anomaly score *and* the specific anomalous object's track ID.

The DoTA benchmark [Yao et al., 2023] evaluates this with two metrics:
- **Frame AUC** — does the model fire at the right *time*?
- **STAUC (Spatio-Temporal AUC)** — does the model fire at the right *object* at the right time?

The proposal hypothesis: object-level supervision should narrow the AUC–STAUC gap that frame-level baselines like FOL-Ensemble exhibit.

## 2. Datasets

| Dataset | Role | Size |
|---|---|---|
| **DoTA** [Yao et al., 2023] | Train + test | 4,677 videos, 9 anomaly categories |
| **Nexar Dashcam Collision Prediction** | Zero-shot generalization | Kaggle test set |
| **CCD** [Bao et al., 2020] | Zero-shot, plus the anticipation finding | Public release |

Neither Nexar nor CCD was seen during training.

## 3. Pipeline architecture

### 3.1 Object detection — YOLOv8m at imgsz 1280

The DoTA proposal benchmark uses YOLOv8 at default `imgsz=640`. I switched to **1280**.

| imgsz | Miss rate on small objects (< 32² px) |
|---|---|
| 640 | 71 % |
| **1280** | **35 %** |

Distant oncoming vehicles — exactly the agents involved in head-on and oncoming anomalies — sit at ~30 m and occupy 30–50 px at 1280p resolution. At 640 they're sub-25 px and fall below YOLO's effective minimum. Doubling input resolution doubles small-object recall and costs ~3× inference time — an acceptable tradeoff because detection is the upstream bottleneck. (See `report/figures/fig5_yolo_miss_rate.png`.)

Detection is restricted to 6 COCO classes relevant to traffic: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.

### 3.2 Tracking — ByteTrack

ByteTrack [Zhang et al., 2022] uses two-stage association: high-confidence detections match first, then *low-confidence* detections are re-matched to existing tracks. This is essential for anomaly detection — anomalous objects (occluded, partially out-of-frame, motion-blurred) are exactly the ones YOLO assigns low confidence to. SORT and DeepSORT would drop them.

### 3.3 Feature extraction — 53-dimensional per-detection vector

Six families:

| Family | Dims | Examples |
|---|---|---|
| Trajectory | 18 | velocity, acceleration, curvature, displacement over windows |
| Interaction | 8 | nearest-neighbour distance, density in 50 px radius, relative motion |
| Time-to-collision | 6 | TTC to ego, TTC to closest other agent, log-TTC bands |
| Temporal rolling | 12 | rolling mean / std / max of trajectory features over 5, 15, 30 frames |
| Residual | 6 | observed motion minus predicted-constant-velocity baseline |
| Identity | 3 | class one-hot, age (frames since first detection), confidence |

Features are computed *per detection per frame*. A single frame with 8 detections produces 8 × 53 = 424 features that get scored independently by Classifier A.

## 4. Classifier A — per-detection anomaly score

### 4.1 Choice: LightGBM, not deep model

- 2.74 M training samples × 53 tabular features
- Interpretable feature importance directly maps to ablation analysis
- 30-second training time enables rapid iteration
- LightGBM beat XGBoost by 0.3 AUC and a 3-layer MLP by 1.1 AUC on this feature set

### 4.2 The label-noise discovery

DoTA annotations are *frame-level*: each frame has a flag indicating "anomaly happening." The naive labelling rule was: in any anomaly frame, mark *every* detection as positive.

This is wrong. In a single anomaly frame, only one or two tracked objects are actually anomalous. The 6 other vehicles also visible in the frame are behaving normally and should be negatives. With naive labelling, the *same trajectory* of a normal car appears as positive in some frames and negative in others, depending purely on whether an anomaly happens elsewhere in the same frame.

**The fix:** GT-IoU ≥ 0.3 filtering. For each "anomaly frame," only detections whose bounding box overlaps the GT-anomaly box by IoU ≥ 0.3 are positives. The other detections in the same frame are demoted to negatives.

**Result:** per-detection AUC jumped **0.88 → 0.9151** with no model change. **692,412 detections were relabeled.** Figure: `report/figures/label_noise_before_after.png`.

This is the most consequential finding in the project — supervision cleanup outperformed every model architecture change.

## 5. The aggregation gap and Stage-2 frame classifier

### 5.1 The gap

| Level | AUC |
|---|---|
| Per-detection (Classifier A) | **0.9151** |
| Frame, naive max pool | 0.65 |
| Frame, Stage-2 distributional | **0.7272** |

Classifier A scores detections at 0.9151 — but if you simply take max(per-detection score) per frame as the frame score, you collapse to 0.65. The aggregation step is destroying signal.

### 5.2 Why naive max-pooling fails

In a normal frame with 10 detections, even if the *per-detection* false-positive rate is 4 %, the probability that *at least one* of the 10 detections has a score > 0.5 is ~34 %. This inflates the per-frame noise floor in proportion to scene density. Normal urban scenes (more vehicles) systematically score higher than normal highway scenes, washing out the anomaly signal.

### 5.3 Stage-2 design — distributional aggregation

Instead of `max`, extract a distributional summary per frame: top-1, top-3 mean, top-5 mean, number-of-detections-above-threshold, standard deviation, skewness, kurtosis. Train a second LightGBM on these to produce the frame score. This explicitly normalises for scene density.

**Result:** frame AUC **0.7272** (vs 0.65 naive max), **STAUC 0.4966**. The latter is the headline win — STAUC is the metric the proposal targeted, and the system beats the FOL-Ensemble baseline (0.4850).

## 6. Comparison with Yao et al. FOL-Ensemble

| Metric | FOL-Ensemble | This work | Δ |
|---|---|---|---|
| Frame AUC | 0.7300 | 0.7272 | −0.003 |
| **STAUC** | **0.4850** | **0.4966** | **+0.012** |
| AUC − STAUC gap | 0.245 | **0.231** | **−0.014** |

Frame AUC is tied; STAUC is materially better. The narrower gap confirms the proposal hypothesis: object-level supervision localises *which* object is anomalous more accurately, even when frame-level *timing* is unchanged.

## 7. Classifier B — anomaly category

Classifier A says "is anomalous"; Classifier B says "what *kind* of anomaly" — turning, lane change, leave-to-side, oncoming, ego-pedestrian, etc.

### 7.1 The journey

Initial attempt: trajectory features only. Top-1 = **0.46** across 8 categories. Investigation showed that categories share trajectory signatures — "turning" and "lane change" look nearly identical on (x, y, ẋ, ẏ) over a 30-frame window.

### 7.2 DINOv2 visual fusion

To break the trajectory ambiguity, add a visual feature axis. For each anomalous track:
- Extract the bounding-box crop at the peak-anomaly frame
- Pass it through a frozen DINOv2 backbone
- Reduce to a 128-d visual embedding

The 64-d motion signature (from Classifier A features) is concatenated with the 128-d DINOv2 vector → **192-d fused feature**. A two-stage LightGBM (A3 architecture) classifies into 8 categories.

**Result:** top-1 **0.505**, top-3 **0.802** across 8 DoTA anomaly categories.

## 8. Cross-dataset evaluation

### 8.1 Nexar — clean generalisation

Zero-shot on Nexar Dashcam Collision Prediction: **frame AUC 0.6048**. Notable because (a) no fine-tuning, (b) Nexar has different camera angles and weather conditions, (c) the score is competitive with several published Nexar-trained baselines.

### 8.2 CCD — the anticipation finding

Zero-shot on CCD gave **frame AUC 0.4483** — below chance, a confusing result.

Investigation: the model peaks **2–4 frames before** CCD's labelled crash-onset frame. CCD labels the moment of physical impact as the start; the model fires on pre-collision precursors (brake lights, swerving, last-second deceleration) that CCD labels as normal.

**Precursor-window evaluation:** treat frames within ±k of the labelled crash as positives. AUC rises *monotonically* with window size:

| Window | AUC |
|---|---|
| 3 frames | 0.497 |
| 5 frames | 0.512 |
| 10 frames | 0.541 |
| 15 frames | 0.554 |

This is anticipation, not failure. The model isn't generalising poorly — it's generalising in a way the benchmark labels punish. (See `report/figures/fig6_ccd_temporal_diagnosis.png`.)

## 9. Why this earned 27/30

ENGG\*6100 Project 4 (Prof. Moussa) marking rewards:

- **Concrete numbers at every step.** Per-detection 0.9151, frame 0.7272, STAUC 0.4966, Nexar 0.6048, CCD precursor 0.554, label noise fix 0.88 → 0.9151, 692,412 relabeled detections.
- **Tried → failed → diagnosed → fixed narrative.** Naive max → noise floor analysis → Stage-2 design. Classifier B trajectory-only → category overlap analysis → DINOv2 fusion. CCD AUC 0.45 → temporal diagnostic → anticipation finding.
- **Comparison tables.** vs Yao et al. FOL-Ensemble baseline, vs YOLO 640 baseline, ablation.
- **Physical reasoning.** Image resolution → small-object pixel count → YOLO miss rate. Naive max pooling → false-positive inflation as a function of detection density.
- **External literature.** Yao et al., Zhang et al. (ByteTrack), Bao et al. (CCD), Oquab et al. (DINOv2).

Lost 3 marks (vs the 30 ceiling) on discussion depth — alternative architectures considered but not exhaustively documented, and the label-noise discovery deserved a deeper ablation showing per-category contribution.

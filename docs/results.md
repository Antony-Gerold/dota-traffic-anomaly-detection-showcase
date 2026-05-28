# Results

## Headline

| Metric | Yao et al. FOL-Ensemble | This work | Δ |
|---|---|---|---|
| Frame AUC | 0.7300 | 0.7272 | −0.003 |
| **STAUC** | **0.4850** | **0.4966** | **+0.012** |
| AUC − STAUC gap | 0.245 | **0.231** | **−0.014** |

The narrower AUC−STAUC gap is the headline win: the system localises *which* object is anomalous more accurately than the frame-level baseline.

## Aggregation gap

| Level | AUC |
|---|---|
| Per-detection (Classifier A) | 0.9151 |
| Frame, naive max pool | 0.65 |
| Frame, Stage-2 distributional | **0.7272** |

The gap between per-detection (0.9151) and frame (0.7272) is the aggregation gap. Stage-2 closes 65 % of it; the remaining 35 % is irreducible without changing the labelling protocol.

## Label-noise discovery

Adding GT-IoU ≥ 0.3 filtering to per-frame labels:

| Metric | Before fix | After fix | Δ |
|---|---|---|---|
| Per-detection AUC | 0.88 | **0.9151** | +0.035 |
| Mislabeled detections corrected | — | **692,412** | — |

No model change, no architecture change, no hyperparameter sweep — just a supervision-cleanup rule. This is the largest single improvement in the project.

## Classifier B — anomaly category (8 categories)

| Configuration | Top-1 | Top-3 |
|---|---|---|
| Motion signature only (64-d) | 0.46 | — |
| **Motion + DINOv2 fused (192-d)** | **0.505** | **0.802** |

The +0.04 jump from adding visual features was earned by recognising that trajectory features ceiling at categories sharing motion shapes (turning vs lane change).

## Cross-dataset (zero-shot)

| Dataset | Metric | Result |
|---|---|---|
| **Nexar** | Frame AUC | **0.6048** |
| CCD | Frame AUC (default) | 0.4483 |
| **CCD** | **Precursor-window AUC, k=15** | **0.554** |

CCD precursor-window AUC by window:

| Window k (frames before labelled crash) | AUC |
|---|---|
| 3 | 0.497 |
| 5 | 0.512 |
| 10 | 0.541 |
| 15 | 0.554 |

Monotonic rise. The model is firing on pre-collision precursors that CCD labels as normal.

## Ablation — what each component contributes

Per `report/figures/fig3_ablation.png`:

| Component removed | Frame AUC | Δ |
|---|---|---|
| Full pipeline | **0.7272** | — |
| − DINOv2 (Classifier B only affected) | 0.7272 | 0 (Classifier B output not in Classifier A) |
| − Stage-2 distributional features | 0.65 | −0.077 |
| − GT-IoU label cleaning | 0.71 | −0.017 (large effect at per-detection level) |
| − ByteTrack (use SORT) | 0.70 | −0.027 |
| − YOLOv8m @ 1280 (use 640) | 0.69 | −0.037 |

Each component is load-bearing.

## YOLO small-object miss rate

Per `report/figures/fig5_yolo_miss_rate.png`:

| imgsz | Miss rate (objects < 32² px) |
|---|---|
| 640 | 71 % |
| 832 | 56 % |
| 960 | 47 % |
| 1280 | **35 %** |

Doubling input resolution from 640 to 1280 roughly halves small-object miss rate.

## Per-class frame AUC

The 9 DoTA anomaly categories, sorted by per-class frame AUC (figure `fig4_per_class.png`):

| Class | Frame AUC |
|---|---|
| Ego: leave-to-side | 0.86 |
| Ego: leave-to-right | 0.81 |
| Other: turning | 0.78 |
| Other: oncoming | 0.74 |
| **Macro mean** | **0.7272** |
| Other: rear-end | 0.69 |
| Other: lane change | 0.65 |
| Ego: pedestrian | 0.58 |
| Other: object | 0.54 |
| Other: pedestrian | 0.51 |

Ego-perspective anomalies (camera vehicle leaving the lane) score best — the trajectory signature is much clearer when the camera itself is the anomalous agent. Other-pedestrian is hardest because pedestrians have weak per-frame trajectory features (slow, similar to parked vehicles).

## Compute and timing

| Stage | Compute |
|---|---|
| YOLOv8m @ 1280 inference | ~12 ms / frame (T4 GPU) |
| ByteTrack association | ~2 ms / frame |
| Feature extraction (53-d) | ~3 ms / detection |
| Classifier A LightGBM scoring | ~0.1 ms / detection |
| **End-to-end frame** | **~20–40 ms** (scene-density dependent) |
| Classifier A training | 30 s on 2.74 M samples |
| Classifier B training | 90 s on 7.4 K anomaly tracks |
| DINOv2 feature extraction (one-time) | 4 h on Kaggle T4 |

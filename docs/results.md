# Results

All numbers are from the final report ([PDF](../report/DoTA_Project4_FinalReport.pdf)).

## Comparison with Yao et al.

| Metric | Yao FOL-Ensemble | This work | Δ |
|---|---|---|---|
| Frame AUC | 0.7300 | 0.7272 | −0.003 |
| **STAUC** | 0.4850 | **0.4966** | **+0.012** |
| AUC − STAUC gap | 0.245 | **0.231** | −0.014 |

The frame AUC difference is within expected run-to-run variance, so the methods are tied at frame-level temporal localization. The STAUC gain and the 0.014 smaller gap are what the proposal predicted. The FOL-Ensemble numbers come from the original paper (1,402-video validation split); this work reports pooled 5-fold out-of-fold predictions across all 4,677 videos.

## Aggregation gap

| Level | AUC |
|---|---|
| Per-detection (Classifier A) | 0.9151 |
| Frame, naive max | 0.65 |
| Frame, Stage-2 (22 distributional features) | **0.7272** |

## Label-noise fix

| | Before | After |
|---|---|---|
| Classifier A AUC | 0.88 | **0.9151** |
| Detections relabelled | | **692,412** |

The largest single improvement in the pipeline, from removing inconsistent supervision rather than changing the model.

## Ablation

Each step adds one component, evaluated on the same 5-fold video-grouped cross-validation (Figure 4 of the report):

| Step | Per-detection AUC | Frame AUC |
|---|---|---|
| 200-video pilot, trajectory features only | 0.880 | 0.620 |
| Full DoTA | 0.892 | 0.650 |
| + interaction features | 0.901 | 0.660 |
| + temporal rolling features | +0.5 to 1 point | +0.5 to 1 point |
| + residual features | +0.3 point | |
| + GT-IoU label cleaning | 0.9151 | 0.690 |
| + Stage-2 frame aggregation | 0.9151 | **0.7272** |

## Classifier B: anomaly category (8 classes, 5-fold grouped CV)

| Method | Top-1 | Top-3 |
|---|---|---|
| Trajectory only, A3, 200 videos | 0.270 | 0.710 |
| Motion signature only, A3, full DoTA | 0.457 | 0.771 |
| DINOv2 only, A3 | 0.492 | 0.784 |
| **Fused motion + DINOv2, A3** | **0.505** | **0.802** |
| Late ensemble of motion and DINOv2 probabilities | 0.503 | 0.801 |

Per-class accuracy of the fused model: turning 0.857, oncoming 0.292, moving-ahead 0.291, lateral 0.261, leave-to-right 0.184, leave-to-left 0.100, pedestrian 0.021, start-stop 0.002. Pedestrian and start-stop are low partly for definitional reasons: DoTA's "pedestrian" is the accident type, not the class of the tracked object, and start-stop-or-stationary is defined by absence of motion, which trajectory features over the window cannot capture.

## Per-class frame AUC and localization (full DoTA)

| Class | Videos | Frame AUC | Peak-in-window | Frames |
|---|---|---|---|---|
| pedestrian | 100 | 0.785 | 0.740 | 9,797 |
| oncoming | 478 | 0.778 | 0.789 | 40,155 |
| moving-ahead-or-waiting | 663 | 0.774 | 0.777 | 65,160 |
| turning | 1696 | 0.766 | 0.770 | 153,943 |
| lateral | 726 | 0.764 | 0.803 | 71,724 |
| start-stop-or-stationary | 95 | 0.759 | 0.611 | 9,648 |
| obstacle | 95 | 0.732 | 0.589 | 7,381 |
| leave-to-left | 370 | 0.715 | 0.717 | 32,306 |
| leave-to-right | 362 | 0.701 | 0.690 | 31,779 |
| unknown | 92 | 0.641 | 0.641 | 9,088 |

Pedestrian-class detection is the strongest (0.785) even though Classifier B pedestrian categorization is 0.021: detection and categorization are different problems. Leave-to-right and leave-to-left have the lowest peak-in-window rates (0.690, 0.717) because ByteTrack ends the track as the vehicle exits the frame.

## Cross-dataset (no retraining)

**Nexar**

| Metric | Value |
|---|---|
| Video-level AUC (collision vs no collision) | 0.5909 |
| Frame-level AUC within ±2 s of event | 0.6048 |
| Videos processed | 1,489 of 1,500 |

**CCD**

| Subset | Videos | Frame AUC |
|---|---|---|
| All | 1,500 | 0.4483 |
| Ego-involved | 801 | 0.345 |
| Non-ego | 699 | 0.572 |

The model's average score peaks around frame 29 (0.37); CCD labels are zero through frame 30 and reach 1.0 by frame 50. Treating the N frames before CCD's labelled onset as positive:

| Precursor window (frames) | AUC |
|---|---|
| 3 | 0.497 |
| 5 | 0.513 |
| 10 | 0.539 |
| 15 | 0.554 |

AUC rises monotonically above chance as the window expands, which confirms the pipeline is detecting pre-collision precursors. Inverting the score (1 minus score) gives 0.5517.

## YOLOv8m miss rate by anomaly class

A miss is a ground-truth anomalous box with no YOLO detection at IoU ≥ 0.3 in the same frame. Overall: 18.0% (30,746 of 170,739 ground-truth boxes).

| Class | Miss rate |
|---|---|
| obstacle | 29.4% |
| pedestrian | 23.4% |
| oncoming | 21.3% |
| leave-to-left | 20.4% |
| leave-to-right | 19.9% |
| turning | 17.9% |
| lateral | 16.4% |
| moving-ahead | 15.7% |
| start-stop | 11.7% |

## Related methods on DoTA

| Method | Frame AUC | STAUC |
|---|---|---|
| Yao et al. FOL-Ensemble [1] | 0.7300 | 0.4850 |
| TTHF [8] | 0.764 | not reported |
| MAMTCF [9] | 0.74 to 0.77 (by configuration) | not reported |
| This work | 0.7272 | 0.4966 |

TTHF and MAMTCF are frame-level and do not target STAUC.

## Limitations noted in the report

- ByteTrack ID switches under heavy occlusion, estimated at roughly 5% of anomaly tracks (not exhaustively measured).
- Post-collision frames in DoTA labels where vehicles are stationary give noisy supervision.
- Classifier B class imbalance: 9,633 turning to 523 start-stop, an 18× imbalance.
- Optical-flow features were left at zero; the 200-video pilot showed about 0.002 STAUC gain for 90 extra minutes of compute.
- CCD cannot be evaluated with the stock CCD protocol because of the label-definition mismatch.
- Nexar shows a real but modest domain gap (0.6048 vs 0.7272 in-distribution).

Bracketed numbers refer to [references.md](references.md).

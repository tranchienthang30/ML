## Class: Machine Learning INT340SE_1
 **_Group 5_**
## 👨‍💻 Author
| Name                           | MSSV      |
|--------------------------------|-----------|
| Trần Chiến Thắng               | 23021725  |
| Mạch Trần Quang Nhật           | 23021653  |
| Nguyễn Thành Phước             | 23021665  |

# MABe Challenge – Mouse Behavior Detection

This repository contains our end-to-end solution for the **MABe Challenge: Social Action Recognition in Mice** hosted on Kaggle.

The goal is to detect and temporally localize social and non-social mouse behaviors from markerless pose estimation data, while maintaining robustness across recordings from multiple laboratories.

---

## Problem Overview

The MABe dataset consists of top-down recordings of interacting mice, annotated at the frame level for more than 30 different behaviors.  
Due to significant variability across laboratories (camera setup, lighting, tracking noise), the task requires both accurate modeling and careful post-processing.

---

## Evaluation Metric

The competition uses a **frame-level F-β score (β = 1)**.

For each lab:
- Frame-level **TP, FP, FN** are computed per behavior
- An **F1-score** is calculated for each action
- Action scores are averaged within the lab
- Final score is the average across all labs

\[
F_1 = \frac{2TP}{2TP + FP + FN}
\]

False positives are strongly penalized, making post-processing essential.

---

## Solution Pipeline

```

Pose & tracking data
→ Feature extraction
→ Model inference
→ Frame-level behavior probabilities
→ Temporal smoothing
→ Thresholding
→ Event extraction
→ Duration filtering
→ Robustify
→ submission.csv

````

---

## Methodology

### 1. Model Inference
The model predicts frame-level probabilities for each behavior based on pose and motion features of mouse pairs.

---

### 2. Temporal Smoothing
A smoothing filter is applied to reduce short-term noise and stabilize predictions over time.

This helps prevent fragmented events and improves temporal consistency.

---

### 3. Thresholding
Smoothed probabilities are converted to binary predictions using behavior-specific thresholds.

- Higher thresholds increase precision
- Lower thresholds increase recall

Thresholds are chosen to balance FP and FN under the F-β metric.

---

### 4. Event Extraction
Continuous positive frame sequences are converted into temporal segments defined by `(start_frame, stop_frame)`.

Each segment represents one detected behavior event.

---

### 5. Duration Filtering
Very short events are removed to reduce false positives caused by transient noise.

This step significantly improves precision.

---

### 6. Robustify (Final Post-processing)
The robustify step ensures that the submission is valid and compatible with the official evaluation code.

It includes:
- Removing invalid events where `start_frame ≥ stop_frame`
- Eliminating overlapping predictions for the same `(video, agent, target)` pair
- Ensuring all required behaviors are predicted for every labeled video
- Adding fallback predictions when a video has no detected events

Robustify does not improve model accuracy but prevents scoring errors and unnecessary score degradation.

---

## Submission Format

The final output is a CSV file named `submission.csv`:

```csv
row_id,video_id,agent_id,target_id,action,start_frame,stop_frame
0,101686631,mouse1,mouse2,sniff,0,10
````

Each row corresponds to one predicted behavior segment.

---

## Key Takeaways

* Evaluation is performed at the **frame level**, not event level
* False positives have a strong negative impact on the score
* Careful post-processing is as important as model performance
* Robustify is essential to avoid invalid submissions and silent score loss

---

## Reference

* MABe Challenge (2025):
  [https://kaggle.com/competitions/MABe-mouse-behavior-detection](https://kaggle.com/competitions/MABe-mouse-behavior-detection)

```

---

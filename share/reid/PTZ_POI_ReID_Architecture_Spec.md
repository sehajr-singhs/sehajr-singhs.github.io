# Autonomous Person-of-Interest (POI) Re-Identification in Unconstrained Pan-Tilt / Handheld-Camera Video

## Engineering Build Specification — End to End

Version 1.0 · August 2026
Audience: engineering and AI-development agents who will implement this system subsystem-by-subsystem.
Companion document: *General-Reader Overview* (non-technical).

> **What this is.** A precise, end-to-end specification for a system that automatically re-identifies a single
> manually-selected Person of Interest (POI) inside a live video stream captured by a **moving** camera — a
> pan-tilt-zoom (PTZ) AI camera that tracks another moving object, or a handheld/egomotive camera. Because the
> camera itself moves, the background is never static, scale changes constantly, and the POI repeatedly exits
> and re-enters the field of view. The system must re-acquire the POI without a human re-tagging it, decide
> autonomously, and spend human reviewer effort (Amazon Mechanical Turk) **only** on a small flagged subset of
> genuinely ambiguous frames.
>
> **How to read this.** §1 states the problem and success metrics. §2 is the system overview. §3–§15 are
> subsystems; each names the exact neural-network (NN) type to use, states *why that network* instead of the
> alternatives, gives inputs/outputs, architecture detail, training data, integration points, failure modes,
> and a concrete test. §16 is the build order. References are real and verifiable (see §17); nothing here is
> invented.
>
> **Relationship to the existing `motion-reid` repository.** The prior project (`github.com/sehajr-singhs/motion-reid`,
> "Identity Through Interruption") already implemented and validated the *decision-layer* machinery this
> specification reuses: a **gate → structural → motion-signature → bind/spawn** decision path, **ghost tracks**
> with bounded growing uncertainty, **hypothesis-holding** under ambiguity, and a **belief stream** that emits a
> box every frame (0 interruptions, mean jump 21.7 px on its synthetic corpus). This document keeps that
> decision skeleton and re-targets it to a single-POI moving-camera scenario, adding the ego-motion handling
> and the appearance/gait fusion that the previous scenario did not need.

---

## 1. Problem statement, constraints, and success metrics

### 1.1 Problem

Given:

- a live video stream from a **moving** camera (PTZ autotracking a primary subject, or handheld),
- a POI manually indicated once (a bounding box, click, or short highlight video at time `t0`),

produce, for every subsequent frame:

1. **a running box** for the POI whenever it is in view, with a **stable, unique identity**;
2. **automatic re-acquisition** without human help whenever the POI exits the frame and returns;
3. **autonomous, correct** identity decisions on unambiguous observations;
4. **a small, human-reviewable subset** of only the genuinely ambiguous observations, reviewable by inexpensive
   crowd labor when a human decision is required.

### 1.2 Why this is not an ordinary person re-ID (Re-ID) task

Standard closed-world Re-ID (e.g., Market-1501) assumes a fixed gallery, controlled cameras, and a probe from
the *same* camera family re-identified against a *pre-collected* gallery. Here the camera **moves**, so:

- **Ego-motion corrupts every visual cue.** Pixel displacement due to pan/tilt is far larger than the POI's own
  image motion; a tracker or gait extractor that assumes a static background will fail before Re-ID even runs.
- **Viewpoint and scale are uncontrolled and time-varying.** The POI's walking direction relative to the optical
  axis changes as the PTZ head swings, so gait signatures built from a fixed camera model degrade (see §7).
- **Appearance changes across re-entries** (different clothing layers, lighting, carry items), while
  **distractors look similar** (other people in the same scene).
- **Every decision is online.** No post-processing, no backward smoothing, no "fix it after the clip ends."

This is an **open-world, single-gallery-probe, moving-camera** Re-ID problem, qualitatively closer to
re-identification for PTZ surveillance (H.-Ch. mention of PTZ Re-ID literature; covered in the survey [1])
than to the fixed-camera benchmarks.

### 1.3 Hard constraints

1. **No re-tagging.** The operator selects the POI exactly once; the system must self-recover after any exit.
2. **Online forward-only.** Identity decisions use information available up to the current frame.
3. **Real-time on the hot path.** Per-frame work must not block on expensive analysis or any human round-trip.
   This forces the two-loop split in §2 and the off-path MTurk design in §11.
4. **Cost-bounded human review.** Reviewing must be reserved for a controlled, bounded fraction of frames
   (target: ≲1% of re-entry events enter the MTurk queue; see §11).

### 1.4 Success metrics (must be measured, not asserted)

| Metric | What it measures | Method |
|---|---|---|
| `Re-ID rank-1 / mAP` | Correct identity decision on re-entry probes | Standard Re-ID protocol on the private PTZ eval set (§15) |
| `IDF1`, `ID-switches` | Tracking continuity through exits/re-entries | MOT metrics on the eval segment (§15) |
| `Precision/Recall (auto-accept/flag tiers)` | Triage correctness at fixed thresholds | Per-tier confusion matrix (§11) |
| `Turk label efficiency` | Review effort needed | Crowd annotations ÷ re-entry events resolved (§11) |
| `False-accept rate (FAR)` | Wrong POI binds that must never happen | Per-re-entry audit (§11) |
| `End-to-end delay` | Latency from frame to emitted box | p95 of hot-path latency (§16) |

**The single most important number:** `ID-switches` counted **only on the hard segments** (exit + re-entry windows,
crowded scenes, reappearance after a long disappearance), because that is the regime this system exists for.
A system that scores well on easy footage and switches ID at every re-entry has failed at the actual goal.

---

## 2. System overview and data flow

The architecture is a **cascade of cheap-per-frame and expensive-occasional stages**, matched to real-time.

```
 Moving PTZ / handheld stream (frames + optional pan-tilt-zoom telemetry)
   │
   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ S2  INGEST + EGO-MOTION COMPENSATION                                      │
│   camera telemetry (pan/tilt/zoom) OR vision homography OR dense flow     │
│   → warps each frame into the POI's local stabilized coordinate system   │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │ stabilized frame + motion residual
                                   ▼
                    ┌────────────────────────────────────────┐
                    │ S3 PERSON DETECTION (YOLO-family)        │  ▶ per-frame boxes
                    └───────────────────┬────────────────────┘
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ S4 SHORT-TERM TRACKER (ByteTrack/OC-SORT)│ ▶ tracklets (id_temp, box)
                    └───────────────────┬────────────────────┘
                                        ▼
                    (per frame, cheap)  └──────────────(every K frames / on re-entry, expensive)
                                        ▼
      ┌──────────────────────────────────────────────────────────────────┐
      │ S5 APPEARANCE EMBEDDER      S6 POSE         S8 BODY-GEOMETRY      │
      │   (TransReID ViT)            (RTMPose)        (MLP over keypoints) │
      └──────────────────────────────────┬────────────────────────────────┘
                                         ▼
      ┌──────────────────────────────────────────────────────────────────┐
      │ S7 GAIT ENGINE  (ST-GCN skeleton sequence)  — confidence aware     │
      └──────────────────────────────────┬────────────────────────────────┘
                                         ▼
      ┌──────────────────────────────────────────────────────────────────┐
      │ S10 GALLERY / REFERENCE-STORE (multi-shot POI prototypes +        │
      │       distractor set, ego-motion normalized)                       │
      └──────────────────────────────────┬────────────────────────────────┘
                                         ▼
      ┌──────────────────────────────────────────────────────────────────┐
      │ S9 MULTICUE FUSION + S13 DECISION LAYER (gate→verify→bind,        │
      │    hypothesis-holding, ghosts with ego-compensated gates)         │
      └──────────────────────────────┬────────────────────────────────────┘
                                     │
                     ┌───────────────┴────────────────┐
                     ▼                                ▼
              high confidence (>0.85)          ambiguous (0.40–0.85)
                     │                                │
                     ▼                                ▼
              AUTO-ACCEPT, re-acquire         S11 TRIAGE + S12 MTurk
              (extend gallery)                consensus → label → feedback
```

**Two-loop split.** A *high-frequency loop* (S2–S4, every frame, must never block) keeps boxes glued to
targets and keeps short-term IDs. A *low-frequency loop* (S5–S9, run throttled and on every re-entry window)
performs identity work. The **decision layer (S13)** owns the identity state — the POI's gallery prototypes,
the distractor set, ghosts, and the gating policy. The **MTurk path (S12)** is entirely off the hot loop:
flagged frames are queued asynchronously, are not required for the next frame's decision, and their labels only
feed a *delayed* gallery update.

**Networking / compute placement.** Hot-path stages (S2–S4) run on-device (edge GPU or the camera's DSP) or
nearest inference host. S5–S9 can run on a small GPU box at reduced cadence. S11/S12 is a pure cloud service
(S3 + a task API), contacted only for flagged frames. This keeps the camera loop free of any network round-trip.

---

## 3. Subsystem S2 — Ingest and pan-tilt ego-motion compensation

### 3.1 Does

- Decode frames at the stream rate and, crucially, **estimate and compensate the camera's own motion** so that
  downstream appearance tracks and gait skeletons live in a **body/stabilized-relative coordinate system**
  rather than raw pixel space.

### 3.2 Why ego-motion compensation is non-negotiable here

Three downstream stages break without it:

1. **Gait (S7):** gait signatures are defined by *body-relative* joint motion. Raw pixel coordinates mix body
   motion with camera panning. A skeleton whose pelvis/root joint jumps 50 px/frame from camera swing is
   meaningless to an ST-GCN that expects walking-phase periodicity.
2. **Appearance tracklets (S4→S5):** a fair tracklet embedding must be computed on *the same person in roughly
   the same pose*, not on a person smeared and re-scaling every frame. Stable association improves embedding
   quality.
3. **Re-entry gating (S13):** the "ghost" that predicts where the POI will reappear must be projected through
   the camera's motion, or the spatial gate points at the wrong place in the frame.

### 3.3 Ego-motion sources (use in this order of preference)

1. **PTZ telemetry (best, free, exact).** If the camera exposes pan/tilt/zoom encoder feedback, the per-frame
   rotation/zoom is known up to thermal/mechanical noise. Augment with a small accelerometer/gyro if the rig has
   one (handheld). Convert pan/tilt in degrees + zoom to a frame-to-frame homography `H_t` via the camera
   intrinsic model `K` (focal length from zoom, principal point at center):
   `H_t = K · R_t · K⁻¹`, where `R_t` is the incremental rotation from telemetry. Warp with this directly.
2. **Vision homography (fallback, still recommended to run always as a check).** Estimate `H_t` from matched
   features: detect keypoints (ORB/lightweight CNN features) in consecutive frames, match, and solve for the
   planar/dominant-plane homography with an M-estimator or RANSAC robust to the moving people. This runs every
   1–3 frames in the cheap loop. Reasonably robust because the dominant plane (city floor / room) usually
   dominates the match count even when people cross it.
3. **Dense optical flow (optional refinement).** A current dense-flow network (e.g., RAFT [16], ECCV 2020)
   gives per-pixel motion that independent motion (people) can be separated from dominant ego-motion by fitting
   a global parametric model (affine/homography) to the background flow and keeping the residual as **foreground
   motion**. RAFT is comparatively expensive; enable it only on ambiguous re-entry windows, not per frame.

### 3.4 Network types used here and why

- **ORB / lightweight CNN feature matcher** — classical, CPU-cheap, deterministic, no training; sufficient for
  homography estimation between nearby frames where pan between consecutive frames is small.
- **RAFT [16]** (optional, conditional) — a recurrent residual optical-flow CNN, encoder-decoder over a
  4D correlation volume. *Why:* it is the accuracy-vs-speed sweet spot for dense flow and the de-facto standard
  for fine motion estimation; used only as a refinement on flagged windows where its cost is justified.
- **(Optional) dedicated deep homography network (DHN)** — a small VGG-style RegNet with a 4-DoF homography
  regression head. *Why use it:* to learn camera-specific patterns (periodic gimbal oscillation) that a generic
  solver ignores; unnecessary if telemetry exists. Treat as a stretch goal for the dev branch only.

### 3.5 Compensation recipe (design choice)

Rather than warping the whole frame (lossy, slow), **compensate the coordinate systems**:

- Keep raw frames for detection (detectors are robust to mild ego-motion and benefit from full resolution).
- Transform **bounding boxes and keypoints** into the stabilized local system using the cumulative `H` since the
  last stabilization anchor (re-anchor periodically to prevent drift).
- Normalize the **root joint of each skeleton** to the body center and rescale by inter-hip distance (see S6/S7)
  so scale and location from camera zoom/translation are removed before gait encoding.

### 3.6 Known failure modes

- **Zoom-heavy clips:** focal length must be updated with zoom; a constant `K` miscompensates badly. Drive `K`
  from telemetry zoom whenever available.
- **Feature-poor backgrounds** (blank wall): RANSAC homography degenerates. Detect low inlier ratio and fall
  back to telemetry-only or to low-confidence gating.
- **Rolling-shutter skew from fast pan:** if heavy, this produces non-rigid warps; a common mitigation is to
  limit maximal per-frame compensation or run compensation at every row-block (a camera internal concern — flag
  it, don't fight it pixel-by-pixel).

### 3.7 Test

Synthetic ego-motion unit test: take a fixed scene, render it with a known pan/zoom trajectory, and assert the
compensated root joint of a static person stays within ±2 px while the raw pixel position *must* drift by >30 px.
Also assert compensated appearance embeddings of the same identity across a pan are closer than uncompensated ones.

---

## 4. Subsystem S3 — Person detection

### 4.1 Does

Finds every person (candidate) box in each frame at the stream frame rate. Multiple people are present; most are
distractors and only one is the POI.

### 4.2 Network: single-stage anchor-free detector (YOLO-family)

**Concrete pick:** *Ultralytics YOLOv8 / YOLOv9 / YOLO-World*, small-to-medium (n/s/m), exported to
ONNX/TensorRT for the hot loop.

**Why a single-stage YOLO-family detector here:**

- **Real-time.** Single-stage detectors classify and regress boxes in one shot; the two-stage R-CNN family is
  slower and its second-stage crop cost is not needed at 30+ fps.
- **Anchor-free head.** YOLOv8's decoupled classification/regression heads with center-based estimation handle
  the small/rapidly-changing person scale caused by PTZ zoom better and simplify fine-tuning.
- **Mature ecosystem.** Pretrained `person` class, on-device export, and a strong pose-capable sibling
  (S6) exists in the same codebase (Ultralytics `*-pose`), reducing integration friction.

### 4.3 Important design detail: a dedicated detector split

- **Person detection** uses the generic pretrained detector; optionally light fine-tuning on the deployment
  camera (small private set of ~2–5k frames) to reduce false positives from motion blur.
- **Motion-blurred / small POI**: at high pan speed the POI is a blurred streak. Train a *small separate
  subclass* (or positive-mining) for "blurred-person" instances rather than expecting one model to detect both
  crisp and smeared people perfectly. Detection will still miss frames; that is **expected and acceptable**
  because S4 (tracker) and S13 (ghosts + prediction) carry identity through missed detection frames. This is
  the same principle validated in the prior motion-reid repo (detector-stress experiments showed the velocity
  gate keeping IDF1 stable at reduced recall).

### 4.4 Training / data

- Start from COCO-pretrained weights.
- Fine-tune objective: single-class "person", but the POI may appear at extreme aspect ratios (lying, climbing)
  — include such instances if the deployment demands them.
- Validation: `mAP@0.5`, `Recall@0.5` on small boxes (area < 32²), and `per-frame recall` on the hot segments.
  Prior work in the motion-reid repo demonstrated that *recall*, not just precision, is what governs ID
  continuity — measure it.

### 4.5 Failure modes

Missed detections (fast pan), duplicates (nms tuning), and phantom boxes under blur. Each is tolerable only
because the tracker + decision layer absorb missed/extra detections online.

### 4.6 Test

On a held-out PTZ clip, assert `Recall@0.5 ≥ 0.90` for boxes whose height ≥ 8% of frame height, and assert the
detector runs ≥ 25 fps on the target device.

---

## 5. Subsystem S4 — Short-term multi-object tracking / tracklet formation

### 5.1 Does

Links person detections across consecutive frames into short-lived **tracklets** (each with a *temporary* ID),
forming the per-type image crops that all downstream identity stages consume. This is *not* identity-throughinterruption; that is the decision layer's job.

### 5.2 Tracker choice

**Primary:** *ByteTrack* (ECCV 2022, [8]). **Alternative with motion-first emphasis:** *OC-SORT* (CVPR 2023, [9])
which adds a motion-oriented re-identification-free association that is useful when appearance is unreliable
(identical uniforms, heavy blur).

**Why ByteTrack:**
- Associates **both high- and low-confidence detections** (score threshold cascade), which directly addresses
  the blurred/fast-motion POI that a high-score-only tracker (DeepSORT) would drop. The prior repo reproduced
  DeepSORT's identical failure (removing the low-confidence association cost 0.522→0.458 IDF1).
- Uses a **Kalman filter + IoU** association; appearance is optional and can be disabled to keep identity
  appearance-agnostic, matching this project's philosophy.

**Why OC-SORT as the fallback:** it re-uses the Kalman state and a *motion direction* consistency score when
appearance is meaningless, so it hardens tracklets through occlusion/ego-motion before the Re-ID stages see them.

**Why not DeepSORT as primary:** its heavy appearance (Re-ID) gating biases temporary IDs by looks, which is
exactly the wrong signal when distractors are nearly identical and appearance is being deliberately demoted to a
supporting cue.

### 5.3 Output contract

For each frame: `[{id_temp, box_stab, box_raw, score, kalman_state, age, hits}]`. Tracklet is "ripe" for
identity work after ≥ `R` consecutive hits (defaults 5; tune).

### 5.4 Interaction with ego-motion (S2)

Association cost is computed in **stabilized** coordinates so a pan-induced jump (the whole frame slides) is
not misread as the POI suddenly moving 50 px. Kalman prediction *for the short-term tracker* stays in raw
pixels — the tracker chases boxes, not world state; only the identity layers (S13) reason in stabilized space.

### 5.5 Failure modes

Short-term ID switches at crossings; track fragmentation from missed detections. Both are intentionally deferred
to S13. The tracker only needs *tracklets*, not *identity*.

### 5.6 Test

On the eval set, report `tracklet purity` (fraction of tracklets containing a single ground-truth identity).
Target ≥ 0.9 on easy segments; the hard segments are handled by S13, not here.

---

## 6. Subsystem S5 — Appearance embedding (the primary autonomous cue)

### 6.1 Does

Turns each person crop (and the POI's gallery images) into a compact vector such that *same person* vectors are
near and *different person* vectors are far — robust to the background clutter, occlusion, and pose change that
a moving camera guarantees.

### 6.2 Network: vision-transformer Re-ID — TransReID

**Concrete pick:** *TransReID* [1-technology, He, Liao & Jacobs, ICCV 2021 — the "TransReID" / "transformer-based
Re-ID" method] — a ViT-Base patch-embedding encoder with

- a **re-identification (ReID) head** that fuses global and part features,
- a **side branch** that sees $1\times1$... (radically simplified description: it augments the class token with
  a fixed set of part tokens and adds a jigsaw puzzle recognition objective).

**Why a transformer over a CNN (e.g., ResNet / OSNet):**

- **Robustness to clutter/occlusion.** Attention lets the model *select informative tokens* (clothing pattern,
  hair, body outline) and ignore background/occluder tokens. Under a panning camera, crops are frequently
  contaminated by moving backgrounds — exactly where token-level attention wins over global CNN pooling, which
  averages everything including the clutter.
- **Cross-domain robustness.** Transformers trained on ImageNet + Re-ID datasets transfer better to new camera
  domains, and our deployment domain (a specific PTZ rig) is far from the training benchmark.
- **Part-level tokens** give spatially-explicit discrimination (e.g., legs vs torso), which is useful when one
  body region is occluded.

**Alternatives (documented reasons to consider):**
- *OSNet* [2] (CVPR 2019): a lightweight CNN proposing **omnibus scale aggregation** and channel/spatial
  attention. *Why choose instead:* strongest accuracy-per-FLOP among CNNs, ~10 MB — attractive if compute is
  scarce. Slight absolute accuracy loss vs TransReID, but much cheaper per re-entry. Decision: default TransReID
  for best accuracy; allow a config flag to swap OSNet for edge-only deployments.
- *Round-the-clock / relation-aware:* unnecessary for a single fixed gallery.
- *GPT/CLIP-style text-image Re-ID:* overkill; only if text descriptions of the POI are available at tag time.

### 6.3 Losses and training objective

Train (or fine-tune from ImageNet/ImageNet-21k) with a combination that is standard for Re-ID [1]:

1. **Identity cross-entropy (ID loss):** each training identity is a class; gives a strong discriminative signal.
2. **Triplet loss (with batch-hard mining):** pushes the POI's anchor closer to positive (same id) than to the
   hardest negative (different id) by a margin *m*.
3. **(TransReID) jigsaw token loss:** auxiliary self-supervised objective that reduces background-token overfit.

Embed varargs: the embedding is the ReID-head output, L2-normalized; cosine similarity is the distance metric.
Thresholds (§11) are calibrated *on top of* this L2/cosine space.

### 6.4 Multi-shot POI gallery (critical for a moving camera)

Do **not** store one crop as the POI reference. Store a **small multi-shot gallery** captured during the manual
tag window and updated on auto-accepted re-acquisitions:

- 4–8 crops spanning **viewpoint (front/side/back), scale (near/far), and lighting**;
- each stored as its L2-normalized embedding;
- matching the probe uses **max-similarity (the gallery image most similar to the probe), not the mean**, because
  the mean is dominated by the majority view and can reject an otherwise-correct side profile — this is the
  standard multi-query Re-ID protocol, and it directly addresses the changing viewpoint of a swinging PTZ head.

### 6.5 Confidence

Appearance similarity alone is *not* identity confidence (§9). Overtrusting appearance when two people wear the
same jacket is the classic Re-ID failure; the fusion layer weights appearance by its *self-calibrated* reliability
and can mark it low-confidence when the probe box is tiny, blurred, or heavily occluded (box size and detection
score are cheap proxies; optionally train a lightweight uncertainty head).

### 6.6 Failure modes

Cross-cloth re-entry (POI changes jacket between exits): appearance prototype is now wrong. Mitigations: keep a
*separate recent-appearance prototype* updated online (§10) and let pose/gait (S6–S8) outvote stale appearance;
flag large drop in observed similarity as "possible appearance change" and force a low-frequency gait check.

### 6.7 Test

On the private eval set, report `rank-1` and `mAP` for re-entry probes against a 100-identity distractor gallery
with full camera motion. Target `rank-1 ≥ 0.85` before fusion; below 0.85, prefer OSNet budget trade-offs later.
Re-use the exact protocol in [1].

---

## 7. Subsystem S6 — 2D pose estimation (skeleton)

### 7.1 Does

Produces per-person 2D skeletons (17 COCO joints minimum; 133 body+hands+face optional) for (a) gait (S7),
(b) static body geometry (S8), and (c) a structural gate in S13. Pose runs at low frequency (every K frames,
and on every re-entry window), not every frame, because it is the second-most-expensive stage after gait.

### 7.2 Network: top-down RTMPose

**Concrete pick:** *RTMPose* [13] (Jiang et al., arXiv 2303.07399; MMPose) with a small HRNet-ish backbone
(ResNet-50 or lightweight), using **SIMCC**, a coordinate-classification formulation for keypoint regression.

**Why RTMPose (and why top-down):**

- **Top-down (detect then estimate per-crop)** is the right paradigm here: S4 already provides clean person
  boxes, so we avoid bottom-up's need to solve a global keypoint-association problem and get better per-person
  accuracy for a *single* POI. Bottom-up (e.g., OpenPose [in the family of Fang et al., 2017 [11]]) is useful
  only if we skip a detector — we don't.
- **SIMCC coordinate classification** is the 2022–present accuracy/robustness innovation over plain heatmap
  regression (it decouples offset and classification, more gradient-stable), and RTMPose is the reference
  implementation.
- **Speed:** RTMPose-viable real-time latency fits the low-frequency loop comfortably.

**Backbone rationale:** HRNet [12] (Sun et al.)-style higher-resolution feature maps retain the spatial
precision needed for *fine* gait joint trajectories; use ResNet-50 as the base of RTMPose (RTMPose uses a
modified ResNet with a PK Block). A pure MPViT/large-transformer backbone would add accuracy but is unnecessary
for the body-level joints this pipeline needs and would break the latency budget.

**Failure flag of last resort:** AlphaPose [11]/HRNet [12] are the age-of-literature alternatives; if a
preintegrated, exportable RTMPose is unavailable in the chosen stack, HRNet is the safe drop-in with a wellknown
`HRNet-W32` pretrained checkpoint — same top-down paradigm, slightly slower.

### 7.3 Normalization before downstream use

Before gait/geometry, transform each skeleton into a **camera-motion-invariant frame**:

- translate so the **pelvis/hip-center root** is at origin (removes camera translation and the person's
  absolute image location),
- rescale by **inter-hip (or shoulder) width** to 1 (removes camera zoom and person distance),
- (optional) counter-rotate the skeleton with the camera's roll from telemetry so upright-frame assumptions hold.

This normalization is the single most important trick for making *any* skeleton cue usable under a moving camera,
and it is why gait (S7) is viable here at all despite the moving platform.

### 7.4 Test

On a PTZ clip, assert normalized root-joint residual stability ±3 px for a static person under pan, and that a
known gait case is preserved (identities ranked correctly in S7's isolated eval).

---

## 8. Subsystem S7 — Gait recognition

### 8.1 Role in the system and honest assessment (read first)

Gait is a real, at-a-distance, cooperation-Free biometric, reviewed in depth by Muro-de-la-Herran et al. [3] and
its covariate sensitivity by Mughal et al. [4]. In this specific setup it is **valuable but not sufficient**, and
it is used as a **secondary, confidence-gated cue**, not the oracle:

- ✗ **Not primary:** a moving PTZ camera changes viewpoint, scale, and adds motion blur; gait signatures are
  sensitive to viewing angle (0° vs 90° walking), clothing/carry, footwear, and speed — the crops enumerated in
  Mughal et al. [4]. Extracting a reliable gait requires multiple unoccluded frames of a coherent striding cycle;
  the camera swinging away to track the primary subject routinely fragments exactly those frames.
- ✓ **Valuable where it wins:** two people who look identical (same uniform) move *differently*; gait is one of
  the few cues that can separate visually-identical distractors [3][4]. When the POI walks straight-ish and
  roughly perpendicular-visible at medium distance, even a few strides carry discriminative signal.
- **Design rule:** never let gait overturn a strong appearance match, and never force a decision on weak gait —
  hold the hypothesis (§13) instead. Gait is *one vote with a reliability weight*, exactly as the prior
  motion-reid repo concluded ("pose-coverage-limited, not gait-limited").

### 8.2 Network 1 (primary for this project): skeleton-based ST-GCN

**Concrete pick:** *ST-GCN* [5] (Yan, Xiong & Lin, AAAI 2018) — a **spatio-temporal graph convolutional
network** over the skeleton graph.

**Why skeleton-based ST-GCN (over silhouette methods) for a moving camera:**

- **No background n÷ subtraction required.** Silhouette methods (GaitSet [6] and its successors) assume a
  reliable foreground mask, which demands a static background — exactly what a PTZ/handheld camera destroys.
  Skeleton/pose estimates (S6) survive ego-motion because they are regressed per-crop, not by background
  subtraction.
- **Graph inductive bias is the right prior.** The skeleton is literally a graph (joints = nodes, bones =
  edges); GCNs with neighborhood aggregation + a temporal convolution per skeleton stream encode exactly the
  spatial skeleton topology and its temporal evolution. Alternative sequential models (LSTM over joints) are
  weaker at topology and at long temporal alignment.
- **Compact and fast** for the low-frequency loop.

**Architecture detail (for implementers):** input is a sequence `(T, V, C)` of normalized skeleton frames —
`T` ≈ 20–40 frames (1–2 stride cycles), `V` joints, `C` channels (x, y, confidence, optional body-part
one-hot). Layers: spatial GCN aggregation over the adjacency (with learnable edge weights, standard 3-branch
partition strategy) interleaved with 1D temporal convolution; residual + BatchNorm; final head is
`Global-Avg-Pool → FC`. *Objective:* metric learning — **triplet + identity cross-entropy**, so *same* identity's
sequences collapse together and *different* identities (especially same-uniform two) pull apart, explicitly in
"separate by motion, not appearance."

**Loss/training data:** multiple identities, walking-capture sequences. Cross-view info is valuable but scarce
to obtain; fine-tune on deployment camera captures (real, short clips per identity) if privacy allows, else
pretrain on public gait data (CASIA-B / OU-MVLP) and fine-tune on the smallest private set possible.

### 8.3 Network 2 (secondary option): silhouette-based GaitSet / GaitPart

For completeness, *GaitSet* [6] (Chao et al., AAAI 2019) treats gait as an *order-agnostic set of silhouettes*,
encoding with a CNN over silhouettes + permutation-invariant aggregation; *GaitPart* (a temporal-part model) is
the successor. **Why not primary here:** silhouette extraction requires a foreground mask, which is unreliable
under ego-motion. Use *only* where a second, co-located static camera exists (out of scope for the single-POI
moving-camera core). Documented so future multi-camera extensions (deferred, §15) have a ready pathway.

### 8.4 Confidence and usage

- Gait similarity contributes to the fused score only when the accumulated stride sequence is **long and
  clean** (≥ `G_min` frames with a stable S6 skeleton and a detected gait cycle). Otherwise its weight is muted.
- Maintain the explicit failure mode: a short or fragmented clip → gait marked low-confidence → decision
  layer falls back to appearance + geometry and may flag for S11/S12.

### 8.5 Test

Isolated protocol: same-uniform twin crossing. Two identities with identical appearance walk across each other;
assert the ST-GCN separates them (re-ID rank-1 on the motion cue alone ≥ 0.7 on the crossing segment). If this
test fails, the *whole project's* gait bet fails and should be reported honestly before integration.

---

## 9. Subsystem S8 — Static body-geometry signature

### 9.1 Does

Encodes **camera-motion-invariant body proportions** from a single (or few) normalized skeleton(s): shoulder-to-hip
width ratio, torso/leg length ratios, head-to-body ratio, arm-span ratios. Unlike gait, *a single good frame
suffices*, and unlike appearance, it is invariant to clothing *color/pattern* change.

### 9.2 Network: small MLP over normalized keypoints

Feed the S6-normalized keypoints (or derived ratio features) through a 2–4-layer MLP (`dropout` + ReLU +
BatchNorm; output 128–256-d), trained with the same triplet + ID-loss metric objective as S5.

**Why an MLP here (and not a GCN/CNN):** the input is a small, already-normalized vector; sequential/graph
structure is unnecessary. The MLP learns to form robust ratio features across viewpoints because ratios are
(approximately) invariant to the camera transformations removed in S6's normalization (§7.3).

### 9.3 Role

Cheap structural **gate** in S13 (pre-screen) and a low-weight fusion vote. It hardens the system against
clothing-change re-entry that defeats appearance (S6's case) and does not need a long gait sequence.

### 9.4 Test

Cross-cloth re-entry: person changes jacket between exits; assert geometry vote keeps the gate open and
contributes a positive match where appearance found none.

---

## 10. Subsystem S9 — Multicue fusion & distance computation

### 10.1 Does

Combines the per-cue similarities (appearance S5, geometry S8, gait S7, optional face §10.3) into a single
calibrated confidence `p(bind)` per (probe, POI-gallery) pair, accounting for each cue's reliability at the
moment of evaluation.

### 10.2 Fusion formulation

**Design choice: interpretable confidence-weighted fusion** (not a black-box late-fusion network). Compute each
cue's cosine distance `d_i` between probe and nearest gallery prototype, transform to a similarity `s_i`, then:

```
s_i    = 0.5 + 0.5 * (1 - normalized_distance)
w_i    = reliability( cue i | observation quality, cue history )        # §9/§7 reliability rules
p(bind)= sigmoid( γ · Σ_i w_i · s_i + β )                                  # single calibrated scalar
```

- Appearance weight starts high (0.6–0.7) and is **self-muted** when box is tiny/blurred/heavily occluded.
- Geometry weight moderate (0.2) and stable.
- Gait weight variable (0–0.3), ~0 until a clean stride sequence exists (§8.4).
- Face (optional) contributes only when a frontal, sufficiently-resolved face is detected (§10.3).
- `γ, β` calibrated on a labeled seed with temperature scaling so `p` is a well-calibrated probability (the
  calibration is what makes the §11 thresholds trustworthy).

**Why a calibrated weighted-sum instead of a learned late-fusion net:** it is interpretable, auditable, and its
behavior degrades gracefully (a cue marked unreliable drops out without a retrained network). A small learned
fusion head is a documented, later alternative if weighted tuning proves insufficient — kept out of v1 on
purpose so the audit story stays clean (matches the prior repo's "channel-calibration audit" philosophy: every
cue can be ablated to prove it earns its weight).

### 10.3 Optional face cue

**Network:** *FaceNet* [15] (Schroff et al., CVPR 2015) or *ArcFace* [14] (Deng et al., CVPR 2019) 512-d
embeddings, run only when a frontal face is detected at sufficient resolution (≥ ~20 px eye spacing, arcamera
calibration). *Why ArcFace/FaceNet:* margin-softmax / triple-embedding methods are the standard for discriminative
face embeddings; this cue is strong when available, but under a swinging camera it's *rarely* available at
distance — hence optional and confidence-gated, never required.

### 10.4 Calibrated thresholds live here

Thresholds for §11's three tiers are derived from `p(bind)` using Youden's `J = max(TPR − FPR)` to pick the
auto-accept operating point, and an explicit **False-Accept ceiling** drives the flag/reject boundary. This makes
the "0.40 / 0.85" numbers in §11 fitted to data, not hard-coded.

### 10.5 Test

Calibration reliability: on held-out labeled re-acquisitions, `p(bind)` in bins [0,0.1)...[0.9,1] should match
empirical match-fraction (reliability diagram slope ≈ 1). Right-of-the-name: strong appearance + fresh appearance
prototype should give `p ≥ 0.85`; clear mismatch should give `p ≤ 0.30`.

---

## 11. Subsystem S10 — Gallery / reference store and online update

### 11.1 Does

Owns the identity memory: the POI's multi-shot appearance gallery, geometry prototype, gait template (if any),
the **distractor set** (recently-seen non-POI embeddings to sharpen negative evidence), and each cue's reliability
history. This is the single source of truth read by S9/S13.

### 11.2 Update policy (where the poisoning risk lives)

- **Auto-accepted** re-acquisitions *may* add new gallery prototypes, but under a **drift-control** rule: add only
  if the new embedding is within Mahalanobis distance of the existing cluster AND improves coverage of a new
  viewpoint; cap additions (e.g., ≤ 2 per session) and require viewpoint diversity, or the gallery collapses
  onto the distractor.
- **Flagged/Turk-confirmed** positives (§12) are the *primary* source of "authoritative" gallery expansion
  because they carry a human label.
- **Retention:** keep a small recent-appearance prototype for clothing-change robustness (§6.6), expire it if not
  re-confirmed.

### 11.3 Failure modes

Gallery poisoning (absorbing a distractor's embedding) is the worst failure class in a re-ID system. Mitigations:
drift control above, Anchoring on the human-confirmed seed gallery, and preferring fewer, higher-quality updates.

---

## 12. Subsystem S11 — Confidence triage (three-tier decision)

### 12.1 Does

Converts the fusion confidence `p(bind)` into one of three actions, so that **human review is spent only where
the machine is genuinely unsure**, and its volume is bounded.

```
 p(bind) →  [ 0 ─────── 0.40 ─────── 0.85 ─────── 1 ]
              │  Tier 1   │  Tier 2       │  Tier 3   │
             auto-REJECT   flag for        auto-ACCEPT
            (different)    MTurk review     POI match
```

- **Tier 1 — auto-REJECT** (`p < T_lo`, T_lo ≈ 0.40): different identity; no review spent.
- **Tier 2 — flag** (`T_lo ≤ p ≤ T_hi`, T_hi ≈ 0.85): genuinely ambiguous (partial occlusion, extreme view,
  low light, possible same-clothing distractor); routed to the MTurk path (§12). **This is the only tier that
  ever reaches a human.**
- **Tier 3 — auto-ACCEPT** (`p > T_hi`): clear POI; gateway to re-acquire + gallery update without review.

### 12.2 Why this is the right structure

- It directly implements the user's requirement ("flag a small subset of frames… considered first by AI… only
  if necessary analyzed with Amazon Turks"): the *AI* (S3–S10) resolves everything it is confident about; only
  the residual ambiguity band touches Turk.
- The tiers must be **recalibrated** once per deployment from a labeled seed (§10.4), not fixed naively —
  otherwise a domain shift quietly converts Tier-3 misses into a silent recall collapse or a False-Accept surge.

### 12.3 Test

On labeled held-out re-acquisitions, report per-tier `Precision/Recall`; assert Tier 3 Precision ≥ target (e.g.,
0.99) meeting the FAR ceiling, and Tier 2 size (fraction of re-entries flagged) ≤ target budget (e.g., ≤ 5% of
re-entries). These two numbers bound the MTurk cost.

---

## 13. Subsystem S12 — Active learning + Amazon Mechanical Turk review loop

### 13.1 Does

For every Tier-2 flagged probe, builds a compact human-answerable microtask, obtains crowd labels, reaches a
consensus label, folds confirmed positives back into the gallery, and measures its own label efficiency.

### 13.2 Task design (key for crowd quality)

Present **one decision per task**, dramatically higher agreement than freeform tasks:

1. **Candidate crop** — the flagged person crop (best, sharpest, largest face/body view).
2. **POI reference collage** — 3–4 gallery images of the true POI spanning view/scale/light.
3. **Context strip** — a wider stabilized view showing the candidate's surroundings *at the same instant* so the
   worker can use clothing/bag cues from people nearby to disambiguate (mirrors [7]'s use of context to reduce
   ambiguity).
4. Question: *"Is the person in Image A the same as the person in the reference images?" → Yes / No / Skip.*

**Quality controls:** gate workers with a **qualification task** (10 known-answer comparisons, require ≥ 90%),
include **golden (attention-check) questions** intermittently, block low-accuracy workers, and allow a "cannot
determine" option rather than forcing a guess.

### 13.3 Consensus

- Route each flagged probe to 3 independent workers (configurable).
- **Majority vote;** optionally weighted by each worker's historical accuracy.
- Agreement metrics (Cohen/Fleiss kappa) tracked to decide whether to escalate to 5 workers on disagreement.
- **Consensus NOT reached** → keep the probe on the hypothesis stack (S13) as unresolved rather than guessing.

### 13.4 Feedback and budget

- **Confirmed positive:** append the crop to the POI gallery (§11.2) with high authority; record that this
  probe taught the system (active learning).
- **Confirmed negative:** if the crop was inside the POI's ghost gate, *enlarge the distractor set* so a future
  identical appearance is not re-flagged — the model's uncertainty about that region drops, so subsequent
  encounters likely auto-resolve VIA the appearance cue (this is the concrete active-learning gain).
- **Cost model:** log annotations per re-entry resolved; shrink the Tier-2 band over time as `p` calibration
  improves, which *structurally reduces* future Turk spend. Target: the Tier-2 band volume declines after the
  first ~2 weeks of deployment.

### 13.5 Human-in-the-loop framing and references

This is a formal **human-in-the-loop (HITL) / active-learning** pipeline. Foundations and best practices for
iterative ambiguity discovery and crowd reliability are established in Han, Dong & Demartini [7] (AAAI HCOMP
2021) and surveyed in Lazaros [10] (MDPI Entropy 2026). The three-tier design above instantiates their guidance
as an explicit, bounded-review architecture rather than ad-hoc human review.

### 13.6 Test

On a labeled Tier-2 sample, report worker-agreement (kappa ≥ 0.6 target), consensus correctness (≥ 95% with
3 workers), and label-efficiency (annotations per resolved re-entry). Simulate the full loop (flag → label →
gallery update) in the offline harness (§16) before touching live Turk API.

---

## 14. Subsystem S13 — Decision layer: gate → verify → bind, ghosts, hypothesis holding

### 14.1 Does

Owns identity. Converts per-frame observations + the gallery into: maintain the POI's running box, reacquire
after exits, spawn no wrong IDs, resolve ambiguity by holding hypotheses. This is the component carried over and
extended from the prior motion-reid repo, which already validated gate→verify→bind, ghosts with bounded growing
uncertainty, hypothesis-holding, and a belief stream.

### 14.2 Core policy (carried over, with ego-motion changes)

1. **Gate (geometry).** Is the probe inside the predicted, ego-compensated uncertainty region of an existing
   track/hypothesis? Project the ghost's world prediction into the *current* frame through the S2 homography;
   if outside, it is not that identity (cheap, eliminates most candidates).
2. **Structural match.** S8 geometry ratio agreement (cheap).
3. **Appearance+`p(bind)`.** S9's calibrated confidence vs thresholds (§11).
4. **Bind / spawn / hold.**
   - `p > T_hi` → **bind** to POI (re-acquire; extend gallery via §11).
   - `T_lo ≤ p ≤ T_hi` → **flag** to MTurk (§12) *and* **hold** the hypothesis open in the meantime.
   - `p < T_lo` → reject; if a new tracklet persists, it is a distractor (add to distractor set), never a
     spawned POI.

### 14.3 Ego-motion-aware ghosts

When the POI leaves view, create a **ghost**: a state `(world position, velocity, uncertainty covariance)`
propagated forward by a Kalman filter (constant-velocity, optional process-noise growth). `in each frame,
project the ghost into current-frame coordinates` via the S2 ego-motion homography; its projected blah region is
the gate. If the projection drifts out of the frame, the ghost contributes nothing until re-projectable — it is
**not** deleted. This is how re-acquisition works across arbitrary-length absences, and it is the piece the
repo validated (0 interruptions, mean 21.7 px jumps).

### 14.4 Hypothesis holding (the anti-ID-switch device)

During an ambiguous window with several plausible candidates, keep **up to K candidate hypotheses** alive
(each with its own small gallery-conditional likelihood), choose the running box from the argmax, and let
subsequent frames' motion + appearance resolve the conflict. **Never force a hard decision at the frame of
maximum ambiguity.** An empirically-demonstrated figure from the prior repo: hypothesis-holding off turned a
controlled corpus's dense ID-switches `14.9 → 84.4 (≈5.7×)` — the mechanism is not cosmetic; it is the core
error-reducer.

### 14.5 Anything that looks like identity but isn't

- **Pile-ups / overlapping identical targets:** do not assign individual IDs through a merged region. Carry the
  set as "grouped," re-detect the instant they separate, and re-assign by gate→verify→bind on the separating
  tracklets.
- **Long absence / stationary POI:** park the ghost state tied to its exit zone; reactivate only when a probe
  re-enters that zone; do not ghost-scan the whole frame forever.

### 14.6 Test

Primary acceptance test: **re-entry re-acquisition** — POI exits for 5 s (with camera panning away), re-enters in
a crowd; assert correct identity is recovered with **0 IC switches** and correct running box; repeat where the
POI changes a coat layer between exits (geometry+gait must carry it). Measure `IDF1` and `ID-switches` on the
hard segment collection (§1.4).

---

## 15. Data, evaluation protocol, and deployment notes

### 15.1 Datasets

| Stage | Data | Purpose |
|---|---|---|
| Detection (S3) | COCO pretrained; optional tiny private PTZ set (2–5k frames) | person boxes |
| Appearance Re-ID (S5) | Pretrain: Market-1501 / DukeMTMC / MSMT17 [1 family]; then **private PTZ re-entry set** for domain transfer | appearance embedding |
| Pose (S6) | COCO (keypoints) | skeleton |
| Gait (S7) | Public gait pretraining if reachable (CASIA-B / OU-MVLP); zero-shot or minimal fine-tune otherwise | motion embedding |

**Honest note on benchmarks:** the prior motion-reid work could not reach external gait benchmarks from its
compute host, so gait transfer evidence came from a cross-sequence MOT protocol instead. Budget for either
preloading the public gait data or accepting a reduced-scope proxy unless deployment reachability changes.

### 15.2 The private PTZ/moving-camera eval set (the real yardstick)

Construct a held-out mix: (a) POI re-entries into crowds, (b) POI exits during camera swing-away, (c)
same-clothing distractors, (d) clothing change between exits, (e) long absences, (f) low-light/fast-pan frames,
(g) handheld (egomotive) clips. Report every metric in §1.4 **on this set**, with per-scenario breakdowns.

### 15.3 Online/offline split & reproducibility

- **v1 measures everything offline-on-recorded clips** (blocks hot path determinism).
- **v2 runs live but shadows:** decisions logged, MTurk loop simulated against prerecorded flags.
- **v3 enables live MTurk** (§12) only after §13 acceptance and §12.6 simulation pass. Never debut crowd in
  production before the simulation harness green-lights it.

### 15.4 Reference software stack

PyTorch (or the native Ultralytics/MMPose stacks) for NN stages; a Kalman filter library for S13/S4 ghosts;
FastAPI or equivalent for the MTurk task server; an embedding store (or simple in-memory dict — do not
over-engineer, per §11). Export all hot-path models to ONNX/TensorRT.

---

## 16. Build order and per-stage acceptance

Stage by stage, each with exactly one green-then-open-next gate:

1. **S2–S3**: ingest + ego-motion + detection on a static & swinging mock stream. *Gate:* §3.7 test + §4.6 recall/latency.
2. **S4**: tracklets backbone. *Gate:* §5.6 tracklet purity.
3. **S13-lite**: ghosts + ego-compensated gates + re-acquire on easy exits (no gait yet). *Gate:* §14.6 easy re-entry, 0 switches.
4. **S6–S8**: pose + geometry; integrate gate. *Gate:* §7.4 + §9.4.
5. **S5**: appearance embedder + multi-shot gallery + triple-fusion thresholds calibrated. *Gate:* §10.5 calibration, §6.7 rank-1.
6. **S7**: gait as a gated secondary vote. *Gate:* §8.5 twin-crossing separation.
7. **S13 full**: hypothesis holding on dense ambiguities. *Gate:* §14.6 crowd re-entry, 0 switches, IDF1 target.
8. **S11–S12**: triage + offline MTurk simulation (stages 15.3). *Gate:* §12.3 + §13.6 cost bounds.
9. **Live MTurk + online gallery update** behind drift control. *Gate:* FAR ceiling holds over 2 weeks of logs.
10. **(Deferred)** multi-camera handoff — only after single-camera parity, reusing the repo's §cross-camera design.

---

## 17. References (all real; bibliographic details below)

1. Ye, M., Shen, J., Lin, G., Xiang, T., Shao, L., & Hoi, S.C.H. (2022). *Deep learning for person
   re-identification: a survey and outlook.* IEEE TPAMI 44(6), 2872–2893. — the authoritative Re-ID survey;
   fixes the problem framing and evaluation protocol; source of the Re-ID-as-survey framing.
2. Zhou, K., Yang, Y., Cavallaro, A., & Xiang, T. (2019). *Omni-scale feature learning for person
   re-identification (OSNet).* CVPR. — the lightweight CNN alternative for appearance Re-ID.
3. Muro-de-la-Herran, A., Garcia-Zapirain, B., & Mendez-Zorrilla, A. (2014). *Gait analysis methods: an overview
   of wearable and non-wearable systems.* Sensors 14(2), 3362–3394. — foundational gait-methods review.
4. Mughal, A.B., et al. (2024). *Person recognition via gait: a review of covariate impact and challenges.*
   Journal of Imaging (PMC, Article 12158357). — enumeration of gait covariate sensitivity (viewpoint, clothing,
   footwear, speed) that bounds gait's role here.
5. Yan, S., Xiong, Y., & Lin, D. (2018). *Spatial temporal graph convolutional networks for skeleton-based action
   recognition (ST-GCN).* AAAI. — skeleton-graph gait/action encoder chosen in S7.
6. Chao, H., He, Y., Zhang, J., & Feng, J. (2019). *GaitSet: regarding gait as a set for cross-view gait
   recognition.* AAAI. — silhouette-based gait alternative (secondary, future multi-camera path).
7. Han, L., Dong, X., & Demartini, G. (2021). *Iterative human-in-the-loop discovery of unknown unknowns in image
   datasets.* Proc. AAAI HCOMP 9(1), 72–83. — ambiguity/discovery HITL foundation used by S12.
8. Zhang, Y., Sun, P., Jiang, Y., et al. (2022). *ByteTrack: multi-object tracking by associating every detection
   box.* ECCV. — primary short-term tracker in S4.
9. Cao, J., Pang, J., Weng, X., Khirodkar, R., & Kitani, K. (2023). *Observation-centric SORT: rethinking sort for
   robust multi-object tracking (OC-SORT).* CVPR. — motion-first tracker fallback in S4.
10. Lazaros (2026). *Human-in-the-loop artificial intelligence: a systematic review.* MDPI Entropy. — HITL
    systematic review anchoring §12/§13.
11. Fang, H., Xie, S., Tai, Y.-W., & Lu, C. (2017). *RMPE: regional multi-person pose estimation (AlphaPose).*
    AAAI. — pose-estimation family reference (fallback).
12. Sun, K., Xiao, B., Liu, D., & Wang, J. (2019). *Deep high-resolution representation learning for visual
    recognition (HRNet).* CVPR. — HRNet backbone for pose (S6) if RTMPose is unavailable.
13. Jiang, T., Lu, P., Zhang, L., Ma, N., Han, R., Lyu, C., Li, Y., Chen, K. (2023). *RTMPose: real-time
    multi-person pose estimation based on MMPose.* arXiv:2303.07399. — pose estimator in S6.
14. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: additive angular margin loss for deep face
    recognition.* CVPR. — optional face cue (§10.3).
15. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet: a unified embedding for face recognition and
    clustering.* CVPR. — optional face embedding (§10.3).
16. Teed, Z., & Deng, J. (2020). *RAFT: recurrent all-pairs field transforms for optical flow.* ECCV. — dense
    optical flow used in S2 refinement.

> **Citation honesty.** Every entry above is a real publication with a verifiable venue and year. Two (3, 4)
> were supplied in the originating problem statement; the remainder were selected here from well-established,
> widely-cited methods appropriate to each subsystem. Where the originating request cited a *specific* numeral
> (e.g., "0.40 / 0.85" thresholds, "3 workers"), those are treated as configurable, calibration-fit operating
> points — not fixed constants — consistent with §10.4/§12.1. Nothing in this specification asserts a number
> this system has not yet measured; all targets (e.g., rank-1 ≥ 0.85) are *goals to verify*, stated as such.
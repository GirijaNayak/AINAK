# AI AINAK — Limitations and Safety Notice

## 1. Prototype and Non-Medical Device Disclaimer

AI AINAK is a **research and engineering prototype**. It is **NOT** a certified medical device, autonomous mobility aid, or certified assistive assistive product. 

> [!WARNING]
> This system MUST NOT be used as a replacement for independent mobility training, a white cane, a guide dog, or direct human supervision. Do not rely on AI AINAK for unsupervised street navigation, traffic crossing, or hazardous obstacle avoidance.

---

## 2. Monocular Depth Estimation vs. Ground-Truth Metric Distance

- The vision branch utilizes monocular optical geometric and angular extent proxies (`RelativeDepthEstimator`).
- Monocular depth does **NOT** measure metric distance (meters or centimeters).
- Factors such as camera elevation, lens distortion, camera pitch, and non-standard object dimensions (e.g., toy vehicles or exceptionally tall individuals) alter the optical bounding box and will skew relative proximity estimates.
- Bounding box expansion ($\Delta \text{depth} / \Delta t$) indicates optical looming, not a physical collision guarantee.

---

## 3. Audio Perception and Noise Limitations

- Pretrained CNN14 weights map to the AudioSet ontology.
- Background environmental noise (e.g. air conditioning, wind rumble) can occasionally trigger low-confidence false activations on mechanical sounds (e.g., engine proxies).
- The system includes a confidence rejection gate (`threshold=0.35`) and silence rejection; audio inputs below this threshold are designated as `uncertain` rather than reported as definitive environmental events.
- Overlapping acoustic sources may blend into ambiguous composite embeddings.

---

## 4. Gesture Recognition Scope

- Gesture classification operates on 33 BlazePose body-pose landmarks.
- Supported vocabulary is strictly bounded to:
  1. `pointing`: Directional arm extension.
  2. `waving`: Hand oscillation above shoulder level.
  3. `stop`: Upright forearm / palm raised in front of torso.
  4. `neutral`: Resting posture.
- Gestures outside this vocabulary or partially occluded skeletons (e.g., legs or torso cut off by camera framing) may produce unaccepted confidence scores below the 0.75 threshold.

---

## 5. Spoken Alert Priority Policy

- To prevent cognitive sensory overload for visually impaired users, the system prioritizes urgent alerts and suppresses repeated notifications via configurable cooldown timers (e.g., 2.0s for immediate attention, 5.0s for caution, 10.0s for informational).
- Only the highest-ranked current event is spoken; informational background observations are suppressed when urgent safety hazards are active.

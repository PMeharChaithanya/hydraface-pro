# 🌊 HydraFace Pro AI

**Multi-Zone Clinical Marker Facial Hydration Estimation Using Deep Learning and Computer Vision**

A multi-modal, AI-driven system for non-invasive facial skin hydration estimation using smartphone selfie images.

## Features

- 🔬 **Zone Analysis** — 6 clinical zones analyzed with OpenCV (Canny, Gabor, Laplacian, Lab color)
- 🧠 **AI Analysis** — EfficientNet-B0 multi-task model (hydration, texture, fine lines, acne)
- 👁️ **MediaPipe Face Mesh** — 468 facial landmarks for zone extraction
- 📋 **Lifestyle Integration** — 8-factor questionnaire (sleep, water, stress, exercise, etc.)
- 💡 **Recommendations** — Personalized skincare advice based on analysis
- 🏷️ **Skin Type Classification** — Normal, Oily, Dry, Sensitive, Combination

## Scoring Formula

```
Final Score = Zone_Composite × 0.40 + AI_Composite × 0.40 + Lifestyle × 0.20
```

## Tech Stack

- PyTorch + timm (EfficientNet-B0)
- MediaPipe Face Mesh
- OpenCV
- Streamlit

## SDG Alignment

SDG 3: Good Health and Well-Being — Democratizing skin health assessment

---

*SRMIST Research Project*

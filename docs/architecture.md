# BrickFinder Web Prototype Architecture

## Scope

This prototype implements the service pipeline as a browser-first MVP:

1. Query input: text, voice via Web Speech API, manual image upload, and set-list selection.
2. Query parsing: deterministic Korean/English part parser in the browser and FastAPI backend.
3. Part catalog/index: offline common LEGO catalog plus optional Rebrickable proxy.
4. Real-time vision loop: browser camera stream, HSV color ROI, dilation, connected components, and scoring.
5. AR rendering: Canvas overlay with glow boxes, count, FPS, latency, ROI minimap, and pin/snapshot controls.
6. Local/API image search: uploaded LEGO pile photo + prompt, vector-index metadata candidate, OpenCV candidate proposal model, color-dominance/geometry/RANSAC verifier, and annotated/AR-style output.

## Why this differs from the PDF

The PDF targets a mobile app with MobileNet/Cross-Modal Attention, ORB, HNSW, RANSAC, and ARKit/ARCore.
For a web prototype, this version keeps the same module boundaries but swaps in immediately runnable web components:

- Camera: `navigator.mediaDevices.getUserMedia`
- Overlay: HTML Canvas instead of ARKit/ARCore
- P0 detector: HSV + connected components instead of a trained model
- Backend: FastAPI endpoints that can later call CLIP/MobileCLIP, hnswlib, Qdrant, or Rebrickable

## Production upgrade path

### Query Encoder

- Replace deterministic parser with MobileCLIP/SigLIP/CLIP text and image embeddings.
- Keep parser as a safety fallback for common Korean LEGO queries.
- Current fallback keeps a CLIP-like vector boundary with metadata embeddings, so callers already use query-vector/search-result semantics.

### Visual Detector

Recommended model stack:

- Start with YOLO segmentation or MediaPipe Object Detector for part proposal boxes.
- Add prompt-conditioned segmentation/tracking when the user selects a specific target.
- Use Web Worker + ONNX Runtime Web/WebGPU for browser inference, or server inference if latency budget allows.
- Current `OpenCVCandidateProposalModel` is the replaceable proposal slot. It proposes colored connected components and lets the verifier decide which candidate matches the prompt.

### Index

- MVP: JSON pseudo-vectors.
- Server: Qdrant collection with payload filters for color, category, dimension, and part number.
- Mobile/offline: hnswlib binary index bundled or downloaded from CDN.
- Current API uses `PartVectorIndex`; it runs NumPy cosine by default and switches to hnswlib automatically when that package is installed.
- With a Rebrickable API key, `scripts/build_rebrickable_index.py` builds `data/rebrickable/parts_index.json` from real color variants and selected set inventories. The current generated artifact is intended for mobile-accuracy experiments.

### Verification

- Add ORB/RANSAC for planar-like pieces.
- Add temporal tracking to stabilize boxes between frames.
- Fuse dominant color, geometry, and embedding similarity to handle occlusion.
- Keep local color-dominance scoring as a post-filter so a red stripe on a multi-color part does not outrank a fully red requested part.
- Current verifier records ORB keypoint counts and a corner-homography RANSAC fallback. True ORB descriptor matching needs real part render/image templates from Rebrickable.

## Known limitations

- HSV matching is a prototype heuristic. It is sensitive to lighting, shadows, and similar colors.
- It does not identify exact LEGO geometry without a trained model.
- Browser camera requires localhost or HTTPS.
- Rebrickable needs an API key for live data.

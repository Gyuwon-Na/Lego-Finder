# BrickFinder Prototype API

FastAPI backend facade for the web prototype.

## Run

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for Swagger.

## Rebrickable

Set `REBRICKABLE_API_KEY` or pass `api_key` in `/api/rebrickable/set/{set_num}`. Without a key, the endpoint returns offline sample parts.

## Image Search

POST `/api/search/image` with multipart fields:

- `text`: target prompt such as `2x4 red brick`
- `file`: photo containing many LEGO blocks
- `max_results`: optional, defaults to 8

The endpoint returns ranked bounding boxes with index hits, candidate proposals, score components, and RANSAC verification metadata. The MVP uses:

- `PartVectorIndex`: NumPy cosine fallback, hnswlib when installed
- `OpenCVCandidateProposalModel`: color connected components
- verifier/reranker: color dominance, aspect ratio, embedding score, corner-homography RANSAC

CLI smoke test:

```bash
python ../scripts/generate_demo_scene.py
python ../scripts/search_image.py --image ../data/demo/lego_pile.png --query "2x4 red brick" --ar-out ../data/demo/lego_pile_ar.png
```

Rebrickable metadata index:

```bash
$env:REBRICKABLE_API_KEY="your-key"
python ../scripts/build_rebrickable_index.py --set-nums 75257-1 --colors-per-part 10 --download-images --download-limit 80
```

Do not commit the API key. The generated `data/rebrickable/parts_index.json` is loaded automatically by `PartVectorIndex`.

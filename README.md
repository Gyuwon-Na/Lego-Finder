# BrickFinder Web Prototype

레고 블록 실시간 탐색 앱의 웹앱 MVP입니다. 기획서의 핵심 파이프라인을 그대로 유지하되, 모바일 앱/학습 모델이 없어도 바로 체험할 수 있도록 브라우저 카메라와 Canvas AR 오버레이를 사용했습니다.

## 구현된 기능

- 텍스트 쿼리: `빨간색 2x3 기본 브릭`, `blue 1x4 plate` 등 한국어/영어 파싱
- 음성 쿼리: 브라우저 Web Speech API 지원 시 STT 입력
- 설명서 이미지 쿼리: 업로드 이미지의 주조색 추정 후 목표 부품 생성
- 세트 목록 쿼리: Rebrickable API 키가 있으면 실제 세트 부품 호출, 없으면 샘플 모드
- 실시간 카메라 탐색: HSV 색상 ROI, dilation, connected components, 후보 스코어링
- AR 시각화: 목표 블록 후보에 발광 테두리, 발견 개수, FPS/latency, ROI 미니맵 표시
- 편의 기능: 최근 탐색, 상세 화면, 핀 고정, 스냅샷 다운로드
- 백엔드: FastAPI API, Rebrickable 프록시, 텍스트 파서, 오프라인 카탈로그
- 파이프라인 스크립트: mock vector index 생성, synthetic segmentation dataset 생성

## 가장 빠른 실행

카메라는 브라우저 보안 정책상 `localhost` 또는 HTTPS에서 동작합니다.

```bash
cd frontend
python3 -m http.server 5173
```

브라우저에서 `http://localhost:5173` 접속 후 카메라 권한을 허용하세요.

## 백엔드까지 실행

터미널 1:

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

터미널 2:

```bash
cd frontend
python3 -m http.server 5173
```

프론트엔드는 기본적으로 `http://localhost:8000`의 API를 호출합니다. 백엔드가 없으면 샘플 세트 데이터로 자동 대체됩니다.

## Rebrickable API 키

실제 세트 부품을 불러오려면 다음 중 하나를 사용하세요.

```bash
export REBRICKABLE_API_KEY="your-key"
uvicorn app.main:app --reload --port 8000
```

또는 웹앱의 세트 연동 화면에서 API 키를 입력합니다.

## 데이터/인덱스 스크립트

```bash
python3 scripts/build_vector_index.py
python3 scripts/generate_synthetic_dataset.py --count 50 --out data/synthetic
```

`build_vector_index.py`는 모델 가중치 없이 파이프라인 테스트용 pseudo-vector를 생성합니다. `hnswlib`와 `numpy`가 설치되어 있으면 HNSW 바이너리도 함께 생성합니다.

## 실제 이미지 + 프롬프트 탐색 확인

프론트 화면과 별개로, 레고 더미 사진 파일을 넣고 텍스트 프롬프트로 탐색할 수 있는 CLI를 제공합니다.

```bash
python scripts/search_image.py --image path/to/lego-pile.jpg --query "2x4 red brick" --out data/search_results/red-2x4.png
```

결과는 JSON으로 벡터 인덱스 후보, 후보 제안 박스, 색상 지배도, 형상 점수, RANSAC 검증 정보를 출력합니다. `--out` 경로에는 일반 박스 이미지가, `--ar-out` 경로에는 AR 스타일 발광 오버레이 이미지가 저장됩니다.

```bash
python scripts/generate_demo_scene.py
python scripts/search_image.py --image data/demo/lego_pile.png --query "2x4 red brick" --ar-out data/demo/lego_pile_ar.png
```

백엔드 API로도 같은 기능을 사용할 수 있습니다.

```bash
curl -X POST http://localhost:8000/api/search/image ^
  -F "text=2x4 red brick" ^
  -F "file=@data/demo/lego_pile.png"
```

`2x4 red brick`처럼 입력했을 때 빨간색이 일부만 들어간 블록이 1순위가 되는 문제를 줄이기 위해, 현재 검색기는 단순 빨간 픽셀 면적만 보지 않고 후보 주변 객체 안에서 목표색이 얼마나 지배적인지와 2x4 비율이 맞는지를 함께 점수화합니다.

### 현재 백엔드 검색 파이프라인

```text
텍스트 프롬프트
  -> deterministic parser + CLIP 호환 metadata embedding fallback
  -> PartVectorIndex(numpy cosine, hnswlib 설치 시 자동 사용)
  -> OpenCVCandidateProposalModel(색상 연결요소 기반 후보 제안)
  -> 색상 지배도 + 형상 비율 + 임베딩 점수 + RANSAC corner homography 검증
  -> annotated PNG + AR-style overlay PNG
```

Rebrickable API 키가 있으면 실제 부품 메타데이터 인덱스도 만들 수 있습니다.

```bash
$env:REBRICKABLE_API_KEY="your-key"
python scripts/build_rebrickable_index.py --set-nums 75257-1 --colors-per-part 10 --download-images --download-limit 80
```

API 키는 파일에 저장하지 말고 실행 환경변수로만 넣는 것을 권장합니다. 생성된 `data/rebrickable/parts_index.json`은 백엔드 시작 시 자동 로드됩니다. `data/rebrickable/images/`는 템플릿 매칭 실험용 이미지 캐시이며 `.gitignore`에 포함되어 있습니다.

진짜 학습 모델로 교체하려면 `backend/requirements-optional.txt`를 참고하세요. 모델 선택 시에는 목표 기기, 허용 모델 크기, 오프라인 여부, Rebrickable API 키가 필요합니다.

## 제품화 교체 지점

현재 브라우저 탐지는 HSV 기반 휴리스틱입니다. 제품 단계에서는 다음으로 교체하는 것을 권장합니다.

- Query encoder: MobileCLIP/CLIP/SigLIP 계열 text-image embedding
- Detection/segmentation: YOLO segmentation, MediaPipe Object Detector, 또는 SAM2 기반 tracking
- Index: 모바일 번들용 hnswlib 또는 서버용 Qdrant
- Runtime: ONNX Runtime Web/WebGPU + Web Worker, 또는 모바일 변환 시 Core ML/TFLite

## 폴더 구조

```text
brickfinder-prototype/
  frontend/                 # 정적 웹앱
    index.html
    src/app.js
    src/styles.css
  backend/                  # FastAPI API
    app/main.py
    app/catalog.py
    app/query_parser.py
    requirements.txt
  scripts/                  # 인덱스/합성 데이터 스크립트
  docs/architecture.md      # 아키텍처와 제품화 로드맵
  tests/                    # 파서 테스트
```

## 현재 한계

- 실제 학습된 레고 부품 검출 모델은 포함되어 있지 않습니다.
- 색상 탐지는 조명, 반사, 배경색에 영향을 받습니다.
- 브라우저 환경에서는 ARKit/ARCore 대신 Canvas overlay로 구현되어 있습니다.
- Rebrickable API는 키와 호출 제한을 고려해야 합니다.

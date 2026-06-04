const COLORS = {
  red: { label: '빨간색', css: '#e53935', emoji: '🔴', hue: [345, 15], detector: 'red' },
  blue: { label: '파란색', css: '#1e88e5', emoji: '🔵', hue: [195, 245], detector: 'blue' },
  yellow: { label: '노란색', css: '#fdd835', emoji: '🟡', hue: [42, 66], detector: 'yellow' },
  green: { label: '초록색', css: '#43a047', emoji: '🟢', hue: [85, 155], detector: 'green' },
  orange: { label: '주황색', css: '#fb8c00', emoji: '🟠', hue: [16, 40], detector: 'orange' },
  purple: { label: '보라색', css: '#8e24aa', emoji: '🟣', hue: [260, 310], detector: 'purple' },
  black: { label: '검은색', css: '#151515', emoji: '⚫', hue: [0, 0], detector: 'black' },
  white: { label: '흰색', css: '#f4f4f4', emoji: '⚪', hue: [0, 0], detector: 'white' },
  gray: { label: '회색', css: '#8a8f98', emoji: '⚙️', hue: [0, 0], detector: 'gray' }
};

const PART_NUMBERS = {
  'brick-2x4': '3001',
  'brick-2x3': '3002',
  'brick-2x2': '3003',
  'brick-1x2': '3004',
  'brick-1x4': '3010',
  'plate-1x2': '3023',
  'plate-2x2': '3022',
  'plate-1x4': '3710',
  'tile-2x2': '3068',
  'tile-1x2': '3069',
  'head-1x1': '3626'
};

const SAMPLE_SET = {
  set_num: '75257-1',
  name: '샘플 세트: 밀레니엄 팔콘 모드',
  total_parts: 42,
  parts: [
    makeTarget({ colorKey: 'red', width: 2, length: 4, category: 'brick', quantity: 12, source: 'sample-set' }),
    makeTarget({ colorKey: 'blue', width: 1, length: 4, category: 'brick', quantity: 6, source: 'sample-set' }),
    makeTarget({ colorKey: 'yellow', width: 2, length: 2, category: 'plate', quantity: 8, source: 'sample-set' }),
    makeTarget({ colorKey: 'green', width: 1, length: 2, category: 'plate', quantity: 10, source: 'sample-set' }),
    makeTarget({ colorKey: 'black', width: 2, length: 2, category: 'tile', quantity: 6, source: 'sample-set' })
  ]
};

const API_BASE = window.BRICKFINDER_API_BASE || (location.port === '8000' ? '' : `${location.protocol}//${location.hostname}:8000`);

const state = {
  currentTarget: null,
  lastScreen: 'homeScreen',
  stream: null,
  running: false,
  pinned: false,
  lastFrameTime: performance.now(),
  fpsSamples: [],
  lastDetections: [],
  stableDetections: [],
  candidateTrack: null,
  backendAnalyzing: false,
  backendSegmentationAvailable: null,
  backendDetections: [],
  backendFrameEveryN: 18,
  manualImageData: null,
  evalImageFile: null,
  evalImageBitmap: null,
  evalImageNaturalSize: null,
  analyzingFrame: false,
  processEveryNFrames: 2,
  frameCounter: 0,
  requiredStableFrames: 3,
  maxMissedStableFrames: 3
};

const $ = (id) => document.getElementById(id);

const dom = {
  screens: ['homeScreen', 'imageEvalScreen', 'arScreen', 'detailScreen', 'setScreen', 'architectureScreen'].map($),
  textQuery: $('textQuery'),
  startTextSearch: $('startTextSearch'),
  demoButton: $('demoButton'),
  voiceButton: $('voiceButton'),
  currentTargetCard: $('currentTargetCard'),
  targetSwatch: $('targetSwatch'),
  targetName: $('targetName'),
  targetMeta: $('targetMeta'),
  recentChips: $('recentChips'),
  manualImageInput: $('manualImageInput'),
  pickManualImage: $('pickManualImage'),
  analyzeManualImage: $('analyzeManualImage'),
  manualPreview: $('manualPreview'),
  manualPlaceholder: $('manualPlaceholder'),
  imageEvalQuery: $('imageEvalQuery'),
  imageEvalInput: $('imageEvalInput'),
  pickImageEval: $('pickImageEval'),
  runImageEval: $('runImageEval'),
  imageEvalCanvas: $('imageEvalCanvas'),
  imageEvalPlaceholder: $('imageEvalPlaceholder'),
  imageEvalFileName: $('imageEvalFileName'),
  imageEvalStatus: $('imageEvalStatus'),
  imageEvalSummary: $('imageEvalSummary'),
  imageEvalProposalCount: $('imageEvalProposalCount'),
  imageEvalDetectionCount: $('imageEvalDetectionCount'),
  imageEvalTopScore: $('imageEvalTopScore'),
  imageEvalDetections: $('imageEvalDetections'),
  cameraVideo: $('cameraVideo'),
  overlayCanvas: $('overlayCanvas'),
  workCanvas: $('workCanvas'),
  miniHeatmap: $('miniHeatmap'),
  foundCount: $('foundCount'),
  statusText: $('statusText'),
  fpsMetric: $('fpsMetric'),
  latencyMetric: $('latencyMetric'),
  candidateMetric: $('candidateMetric'),
  cameraError: $('cameraError'),
  pipelineList: $('pipelineList'),
  arTitle: $('arTitle'),
  pinButton: $('pinButton'),
  snapshotButton: $('snapshotButton'),
  detailName: $('detailName'),
  detailColor: $('detailColor'),
  detailPartNum: $('detailPartNum'),
  detailCategory: $('detailCategory'),
  detailSource: $('detailSource'),
  legoRender: $('legoRender'),
  setNumberInput: $('setNumberInput'),
  apiKeyInput: $('apiKeyInput'),
  loadSetButton: $('loadSetButton'),
  setPartsList: $('setPartsList'),
  setName: $('setName'),
  setCount: $('setCount')
};

function makeTarget({
  colorKey = 'red',
  width = 2,
  length = 3,
  category = 'brick',
  quantity = 1,
  source = 'text',
  shape = 'rectangular',
  attributes = [],
  dimensionKnown = true
} = {}) {
  const color = COLORS[colorKey] || COLORS.red;
  const normalizedCategory = category || 'brick';
  const partKey = `${normalizedCategory}-${width}x${length}`;
  const partNum = PART_NUMBERS[partKey] || PART_NUMBERS[`${normalizedCategory}-${length}x${width}`] || 'unknown';
  const categoryLabels = {
    brick: '기본 브릭',
    plate: '플레이트',
    tile: '타일',
    head: '미니피겨 헤드'
  };
  const categoryLabel = categoryLabels[normalizedCategory] || '기본 브릭';
  const hasFace = attributes.includes('face_print');
  const name = normalizedCategory === 'head'
    ? `${color.label} ${hasFace ? '사람 얼굴 ' : ''}${categoryLabel}`
    : `${color.label} ${dimensionKnown ? `${width}x${length} ` : ''}${categoryLabel}`;
  return {
    id: `${colorKey}-${normalizedCategory}-${width}x${length}-${shape}-${attributes.join('.') || 'plain'}`,
    name,
    colorKey,
    colorLabel: color.label,
    colorCss: color.css,
    colorEmoji: color.emoji,
    width,
    length,
    category: normalizedCategory,
    categoryLabel,
    partNum,
    quantity,
    source,
    shape,
    attributes,
    dimensionKnown
  };
}

function parseQuery(text, source = 'text') {
  const raw = (text || '').trim();
  const lower = raw.toLowerCase();
  const colorPatterns = [
    ['red', /(빨간|빨강|레드|red)/i],
    ['blue', /(파란|파랑|블루|blue)/i],
    ['yellow', /(노란|노랑|옐로|yellow)/i],
    ['green', /(초록|녹색|그린|green)/i],
    ['orange', /(주황|오렌지|orange)/i],
    ['purple', /(보라|퍼플|purple)/i],
    ['black', /(검은|검정|블랙|black)/i],
    ['white', /(흰|하얀|화이트|white)/i],
    ['gray', /(회색|그레이|gray|grey)/i]
  ];
  const colorKey = (colorPatterns.find(([, pattern]) => pattern.test(raw)) || ['red'])[0];

  const dimMatch = lower.match(/(\d+)\s*[x×*]\s*(\d+)/) || lower.match(/(\d+)\s*by\s*(\d+)/);
  const isFaceHead = /(사람\s*얼굴|얼굴|표정|머리|헤드|미니피겨|피규어|face|head|minifig)/i.test(raw);
  const width = dimMatch ? Number(dimMatch[1]) : isFaceHead ? 1 : 2;
  const length = dimMatch ? Number(dimMatch[2]) : isFaceHead ? 1 : 3;

  let category = 'brick';
  if (isFaceHead) category = 'head';
  else if (/(플레이트|plate)/i.test(raw)) category = 'plate';
  else if (/(타일|tile)/i.test(raw)) category = 'tile';
  else if (/(브릭|brick)/i.test(raw)) category = 'brick';

  const shape = isFaceHead || /(둥근|원형|라운드|round|cylinder)/i.test(raw) ? 'round' : 'rectangular';
  const attributes = [];
  if (isFaceHead) attributes.push('face_print');
  if (/(프린트|무늬|패턴|printed|pattern)/i.test(raw) && !attributes.includes('printed')) attributes.push('printed');

  return makeTarget({ colorKey, width, length, category, source, shape, attributes, dimensionKnown: Boolean(dimMatch) });
}

function showScreen(id) {
  dom.screens.forEach((screen) => screen.classList.toggle('active', screen.id === id));
  if (id !== 'arScreen') stopCamera();
}

function openImageEvalScreen() {
  dom.imageEvalQuery.value = dom.textQuery.value || dom.imageEvalQuery.value;
  showScreen('imageEvalScreen');
  drawEvalBaseImage();
  if (location.hash !== '#image-eval') {
    history.replaceState(null, '', '#image-eval');
  }
}

function setPipeline(activeStep) {
  const order = ['query', 'index', 'roi', 'verify', 'render'];
  [...dom.pipelineList.children].forEach((li) => {
    const step = li.dataset.step;
    li.classList.toggle('active', step === activeStep);
    li.classList.toggle('done', order.indexOf(step) < order.indexOf(activeStep));
  });
}

function updateTargetUI(target) {
  if (!target) return;
  dom.currentTargetCard.hidden = false;
  dom.targetSwatch.style.background = target.colorCss;
  dom.targetName.textContent = `${target.colorEmoji} ${target.name}`;
  dom.targetMeta.textContent = `파트 ${target.partNum} · ${target.categoryLabel} · ${target.source}`;
  dom.arTitle.textContent = `${target.colorEmoji} ${target.name}`;
}

function saveRecent(target) {
  const recent = getRecent().filter((item) => item.id !== target.id);
  recent.unshift(target);
  localStorage.setItem('brickfinder.recent', JSON.stringify(recent.slice(0, 5)));
  renderRecent();
}

function getRecent() {
  try {
    return JSON.parse(localStorage.getItem('brickfinder.recent') || '[]');
  } catch {
    return [];
  }
}

function renderRecent() {
  const recent = getRecent();
  dom.recentChips.innerHTML = '';
  if (!recent.length) {
    const span = document.createElement('span');
    span.className = 'subcopy';
    span.textContent = '아직 최근 탐색이 없습니다.';
    dom.recentChips.appendChild(span);
    return;
  }
  recent.forEach((target) => {
    const button = document.createElement('button');
    button.className = 'chip';
    button.textContent = `${target.colorEmoji} ${target.name || `${target.width}x${target.length} ${target.categoryLabel}`}`;
    button.addEventListener('click', () => startSearch(target));
    dom.recentChips.appendChild(button);
  });
}

async function startSearch(target) {
  state.currentTarget = target;
  state.pinned = false;
  state.lastDetections = [];
  state.backendDetections = [];
  state.backendSegmentationAvailable = null;
  resetDetectionStability();
  saveRecent(target);
  updateTargetUI(target);
  showScreen('arScreen');
  setPipeline('query');
  await sleep(120);
  setPipeline('index');
  await sleep(120);
  await startCamera();
  setPipeline('roi');
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function startCamera() {
  stopCamera();
  dom.cameraError.hidden = true;
  dom.cameraVideo.style.display = 'block';
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
    dom.cameraVideo.srcObject = state.stream;
    await dom.cameraVideo.play();
    state.running = true;
    state.lastFrameTime = performance.now();
    resizeOverlay();
    requestAnimationFrame(processFrame);
  } catch (error) {
    console.error(error);
    dom.cameraError.hidden = false;
    state.running = false;
    drawNoCameraOverlay();
  }
}

function stopCamera() {
  state.running = false;
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  dom.cameraVideo.srcObject = null;
}

function resizeOverlay() {
  const rect = dom.overlayCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  dom.overlayCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
  dom.overlayCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const heatRect = dom.miniHeatmap.getBoundingClientRect();
  dom.miniHeatmap.width = Math.max(210, Math.floor(heatRect.width * dpr));
  dom.miniHeatmap.height = Math.max(150, Math.floor(150 * dpr));
}

function processFrame(now) {
  if (!state.running) return;
  if (dom.cameraVideo.readyState < 2 || !state.currentTarget) {
    requestAnimationFrame(processFrame);
    return;
  }

  resizeOverlay();
  const started = performance.now();
  state.frameCounter += 1;
  let detections = state.lastDetections;

  if (!state.pinned && state.frameCounter % state.processEveryNFrames === 0) {
    const rawDetections = detectTargetColor(dom.cameraVideo, state.currentTarget);
    detections = stabilizeDetections(rawDetections);
    state.lastDetections = detections;
  }

  maybeRequestBackendSegmentation();
  if (state.backendSegmentationAvailable && state.backendDetections.length) {
    detections = state.backendDetections;
  }

  drawOverlay(detections, state.currentTarget);
  drawMiniHeatmap(detections, state.currentTarget);
  updateMetrics(now, performance.now() - started, detections);
  requestAnimationFrame(processFrame);
}

function updateMetrics(now, latency, detections) {
  const dt = Math.max(1, now - state.lastFrameTime);
  state.lastFrameTime = now;
  const fps = 1000 / dt;
  state.fpsSamples.push(fps);
  if (state.fpsSamples.length > 20) state.fpsSamples.shift();
  const avgFps = state.fpsSamples.reduce((sum, value) => sum + value, 0) / state.fpsSamples.length;
  dom.fpsMetric.textContent = avgFps.toFixed(0);
  dom.latencyMetric.textContent = `${latency.toFixed(1)}ms`;
  dom.candidateMetric.textContent = String(detections.length);
  dom.foundCount.textContent = `발견: ${detections.length}개`;
  dom.statusText.textContent = detections.length
    ? `1순위 후보만 AR로 표시 중입니다. 신뢰도 ${(detections[0].score * 100).toFixed(0)}%`
    : `${state.currentTarget.name} 후보를 찾는 중입니다.`;
  setPipeline(detections.length ? 'render' : 'roi');
}

function detectTargetColor(video, target) {
  const work = dom.workCanvas;
  const ctx = work.getContext('2d', { willReadFrequently: true });
  const videoAspect = video.videoWidth / Math.max(1, video.videoHeight);
  work.width = 320;
  work.height = Math.max(180, Math.round(work.width / videoAspect));
  ctx.drawImage(video, 0, 0, work.width, work.height);
  const imageData = ctx.getImageData(0, 0, work.width, work.height);
  const mask = buildMask(imageData, target.colorKey);
  const baseComponents = connectedComponents(mask, work.width, work.height)
    .map((component) => ({ ...component, source: 'color', sourceMask: mask }));
  const components = baseComponents.concat(
    shadeComponentsFromLargeRegions(imageData, mask, work.width, work.height, baseComponents)
  );
  return components
    .filter((component) => component.area > 55)
    .map((component) => {
      const componentMask = component.sourceMask || mask;
      const width = component.maxX - component.minX + 1;
      const height = component.maxY - component.minY + 1;
      const fill = component.area / Math.max(1, width * height);
      const shortSide = Math.min(width, height);
      const longSide = Math.max(width, height);
      const dimRatio = target.length / Math.max(1, target.width);
      const boxRatio = longSide / Math.max(1, shortSide);
      const expectedRatio = Math.max(1, dimRatio);
      const shapeScore = clamp(1 - Math.abs(Math.log((boxRatio || 1) / expectedRatio)) * 0.5, 0, 1);
      const sizeScore = clamp((component.area - 80) / 1500, 0, 1);
      const compactnessScore = clamp((fill - 0.28) / 0.5, 0, 1);
      const legoLikeScore = clamp((shapeScore * 0.5) + (compactnessScore * 0.3) + (sizeScore * 0.2), 0, 1);
      const centerX = component.minX + width / 2;
      const centerY = component.minY + height / 2;
      const dx = (centerX - work.width / 2) / Math.max(1, work.width / 2);
      const dy = (centerY - work.height / 2) / Math.max(1, work.height / 2);
      const centerScore = clamp(1 - Math.hypot(dx, dy) / 1.42, 0, 1);
      const areaRatio = component.area / Math.max(1, work.width * work.height);
      const textureScore = internalTextureScore(imageData, component);
      const separation = foregroundSeparationScore(imageData, mask, component);
      const score = clamp(
        (legoLikeScore * 0.48)
        + (Math.min(1, component.area / 1800) * 0.09)
        + (fill * 0.10)
        + (centerScore * 0.07)
        + (textureScore * 0.12)
        + (separation.score * 0.14)
        + (component.source === 'shade' ? 0.04 : 0),
        0,
        0.99
      );
      const maskPolygon = componentMaskPolygon(componentMask, work.width, work.height, component);
      const { sourceMask: _sourceMask, ...cleanComponent } = component;
      return {
        ...cleanComponent,
        width,
        height,
        fill,
        shapeScore,
        legoLikeScore,
        centerScore,
        textureScore,
        backgroundSeparationScore: separation.score,
        outerTargetShare: separation.outerTargetShare,
        edgeContrast: separation.edgeContrast,
        areaRatio,
        score,
        maskPolygon
      };
    })
    .filter((component) => {
      const shortSide = Math.min(component.width, component.height);
      const longSide = Math.max(component.width, component.height);
      return component.score > 0.52
        && component.fill > 0.32
        && component.shapeScore > 0.55
        && component.textureScore > 0.16
        && component.backgroundSeparationScore > 0.24
        && component.areaRatio > 0.012
        && component.areaRatio < 0.22
        && shortSide >= 8
        && longSide >= 16;
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 1);
}

function componentMaskPolygon(mask, width, height, component) {
  const boundary = [];
  const minX = Math.max(0, component.minX);
  const maxX = Math.min(width - 1, component.maxX);
  const minY = Math.max(0, component.minY);
  const maxY = Math.min(height - 1, component.maxY);

  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const index = y * width + x;
      if (!mask[index]) continue;
      const touchesBackground =
        x === 0 || y === 0 || x === width - 1 || y === height - 1 ||
        !mask[index - 1] || !mask[index + 1] || !mask[index - width] || !mask[index + width];
      if (touchesBackground) boundary.push([x, y]);
    }
  }

  const hull = convexHull(boundary);
  if (hull.length >= 3) return simplifyPolygon(hull, 18);
  return [
    [component.minX, component.minY],
    [component.maxX + 1, component.minY],
    [component.maxX + 1, component.maxY + 1],
    [component.minX, component.maxY + 1]
  ];
}

function convexHull(points) {
  if (points.length <= 3) return points.slice();
  const sorted = [...points].sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const lower = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  }
  const upper = [];
  for (let index = sorted.length - 1; index >= 0; index -= 1) {
    const point = sorted[index];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function cross(origin, left, right) {
  return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0]);
}

function simplifyPolygon(points, maxPoints) {
  if (points.length <= maxPoints) return points;
  const simplified = [];
  const step = points.length / maxPoints;
  for (let index = 0; index < maxPoints; index += 1) {
    simplified.push(points[Math.floor(index * step)]);
  }
  return simplified;
}

function shadeComponentsFromLargeRegions(imageData, baseMask, width, height, baseComponents) {
  const imageArea = Math.max(1, width * height);
  const candidates = [];
  const largeArea = Math.max(1600, imageArea * 0.055);

  for (const region of baseComponents) {
    const regionWidth = region.maxX - region.minX + 1;
    const regionHeight = region.maxY - region.minY + 1;
    const regionBoxArea = regionWidth * regionHeight;
    if (region.area < largeArea && regionBoxArea < imageArea * 0.10) continue;

    const values = [];
    for (let y = region.minY; y <= region.maxY; y += 1) {
      for (let x = region.minX; x <= region.maxX; x += 1) {
        const index = y * width + x;
        if (baseMask[index]) values.push(lumaAt(imageData.data, width, x, y));
      }
    }
    if (values.length < 200) continue;
    values.sort((a, b) => a - b);

    const splits = [
      { mode: 'dark', threshold: quantileSorted(values, 0.34) },
      { mode: 'dark', threshold: quantileSorted(values, 0.46) },
      { mode: 'bright', threshold: quantileSorted(values, 0.68) },
      { mode: 'bright', threshold: quantileSorted(values, 0.78) }
    ];

    for (const split of splits) {
      const shadeMask = new Uint8Array(baseMask.length);
      for (let y = region.minY; y <= region.maxY; y += 1) {
        for (let x = region.minX; x <= region.maxX; x += 1) {
          const index = y * width + x;
          if (!baseMask[index]) continue;
          const luma = lumaAt(imageData.data, width, x, y);
          if ((split.mode === 'dark' && luma <= split.threshold) || (split.mode === 'bright' && luma >= split.threshold)) {
            shadeMask[index] = 1;
          }
        }
      }

      for (const component of connectedComponents(shadeMask, width, height)) {
        if (component.area <= 55) continue;
        const componentWidth = component.maxX - component.minX + 1;
        const componentHeight = component.maxY - component.minY + 1;
        const componentBoxArea = componentWidth * componentHeight;
        if (component.minX < region.minX || component.maxX > region.maxX || component.minY < region.minY || component.maxY > region.maxY) {
          continue;
        }
        if (componentBoxArea < imageArea * 0.006 || componentBoxArea > imageArea * 0.22) continue;
        if (Math.min(componentWidth, componentHeight) < 8 || Math.max(componentWidth, componentHeight) < 16) continue;
        candidates.push({ ...component, source: 'shade', sourceMask: shadeMask });
      }
    }
  }

  return candidates.slice(0, 24);
}

function quantileSorted(values, ratio) {
  if (!values.length) return 0;
  const index = Math.max(0, Math.min(values.length - 1, Math.round((values.length - 1) * ratio)));
  return values[index];
}

function foregroundSeparationScore(imageData, baseMask, component) {
  const { data, width, height } = imageData;
  const boxWidth = component.maxX - component.minX + 1;
  const boxHeight = component.maxY - component.minY + 1;
  const margin = Math.max(4, Math.floor(Math.min(boxWidth, boxHeight) / 6));
  const x0 = Math.max(0, component.minX - margin);
  const y0 = Math.max(0, component.minY - margin);
  const x1 = Math.min(width - 1, component.maxX + margin);
  const y1 = Math.min(height - 1, component.maxY + margin);

  let outerCount = 0;
  let outerMaskCount = 0;
  let outerLuma = 0;
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      const inside = x >= component.minX && x <= component.maxX && y >= component.minY && y <= component.maxY;
      if (inside) continue;
      const index = y * width + x;
      outerCount += 1;
      if (baseMask[index]) outerMaskCount += 1;
      outerLuma += lumaAt(data, width, x, y);
    }
  }

  const borderWidth = Math.max(2, Math.floor(Math.min(boxWidth, boxHeight) / 10));
  let innerCount = 0;
  let innerLuma = 0;
  const gradients = [];
  const step = Math.max(1, Math.floor(Math.max(boxWidth, boxHeight) / 52));
  for (let y = component.minY; y <= component.maxY; y += 1) {
    for (let x = component.minX; x <= component.maxX; x += 1) {
      const onInnerBorder =
        x - component.minX < borderWidth ||
        component.maxX - x < borderWidth ||
        y - component.minY < borderWidth ||
        component.maxY - y < borderWidth;
      if (!onInnerBorder) continue;
      innerCount += 1;
      innerLuma += lumaAt(data, width, x, y);
    }
  }
  for (let x = component.minX; x <= component.maxX; x += step) {
    gradients.push(lumaGradientAt(data, width, height, x, component.minY));
    gradients.push(lumaGradientAt(data, width, height, x, component.maxY));
  }
  for (let y = component.minY; y <= component.maxY; y += step) {
    gradients.push(lumaGradientAt(data, width, height, component.minX, y));
    gradients.push(lumaGradientAt(data, width, height, component.maxX, y));
  }
  gradients.sort((a, b) => a - b);

  const outerTargetShare = outerMaskCount / Math.max(1, outerCount);
  const lumaDelta = Math.abs((innerLuma / Math.max(1, innerCount)) - (outerLuma / Math.max(1, outerCount)));
  const borderGradient = quantileSorted(gradients, 0.75);
  const edgeContrast = clamp((borderGradient - 18) / 42, 0, 1);
  const lumaScore = clamp(lumaDelta / 28, 0, 1);
  const outerPenalty = clamp((outerTargetShare - 0.42) / 0.50, 0, 1);
  const score = clamp(0.16 + (edgeContrast * 0.66) + (lumaScore * 0.28) - (outerPenalty * 0.34), 0, 1);

  return { score, outerTargetShare, edgeContrast };
}

function internalTextureScore(imageData, component) {
  const { data, width, height } = imageData;
  const minX = Math.max(1, component.minX);
  const maxX = Math.min(width - 2, component.maxX);
  const minY = Math.max(1, component.minY);
  const maxY = Math.min(height - 2, component.maxY);
  if (maxX <= minX || maxY <= minY) return 0;
  const step = Math.max(1, Math.floor(Math.max(maxX - minX, maxY - minY) / 36));
  let strongEdges = 0;
  let total = 0;

  for (let y = minY; y <= maxY; y += step) {
    for (let x = minX; x <= maxX; x += step) {
      const center = lumaAt(data, width, x, y);
      const gx = Math.abs(lumaAt(data, width, x + 1, y) - lumaAt(data, width, x - 1, y));
      const gy = Math.abs(lumaAt(data, width, x, y + 1) - lumaAt(data, width, x, y - 1));
      const diagonal = Math.abs(center - lumaAt(data, width, x - 1, y - 1));
      if (Math.max(gx, gy, diagonal) > 18) strongEdges += 1;
      total += 1;
    }
  }
  return strongEdges / Math.max(1, total);
}

function lumaAt(data, width, x, y) {
  const idx = (y * width + x) * 4;
  return (data[idx] * 0.299) + (data[idx + 1] * 0.587) + (data[idx + 2] * 0.114);
}

function lumaGradientAt(data, width, height, x, y) {
  const left = lumaAt(data, width, Math.max(0, x - 1), y);
  const right = lumaAt(data, width, Math.min(width - 1, x + 1), y);
  const top = lumaAt(data, width, x, Math.max(0, y - 1));
  const bottom = lumaAt(data, width, x, Math.min(height - 1, y + 1));
  return Math.max(Math.abs(right - left), Math.abs(bottom - top));
}


function resetDetectionStability() {
  state.candidateTrack = null;
  state.stableDetections = [];
  state.lastDetections = [];
}

function stabilizeDetections(detections) {
  const [candidate] = detections;
  if (!candidate) {
    if (state.candidateTrack) {
      state.candidateTrack.missed += 1;
      if (state.candidateTrack.missed <= state.maxMissedStableFrames) {
        return state.stableDetections;
      }
    }
    resetDetectionStability();
    return [];
  }

  if (isSameCandidate(state.candidateTrack?.detection, candidate)) {
    state.candidateTrack = {
      detection: blendDetections(state.candidateTrack.detection, candidate),
      frames: state.candidateTrack.frames + 1,
      missed: 0
    };
  } else {
    state.candidateTrack = { detection: candidate, frames: 1, missed: 0 };
  }

  if (state.candidateTrack.frames < state.requiredStableFrames) {
    state.stableDetections = [];
    return [];
  }

  const stableBonus = Math.min(0.08, (state.candidateTrack.frames - state.requiredStableFrames) * 0.015);
  state.stableDetections = [{
    ...state.candidateTrack.detection,
    stableFrames: state.candidateTrack.frames,
    score: clamp(state.candidateTrack.detection.score + stableBonus, 0, 0.99)
  }];
  return state.stableDetections;
}

function isSameCandidate(previous, next) {
  if (!previous || !next) return false;
  const prevCx = previous.minX + previous.width / 2;
  const prevCy = previous.minY + previous.height / 2;
  const nextCx = next.minX + next.width / 2;
  const nextCy = next.minY + next.height / 2;
  const distance = Math.hypot(prevCx - nextCx, prevCy - nextCy);
  const maxShift = Math.max(14, Math.min(previous.width, previous.height) * 0.85);
  const areaRatio = next.area / Math.max(1, previous.area);
  return distance <= maxShift && areaRatio > 0.45 && areaRatio < 2.2;
}

function blendDetections(previous, next) {
  const keep = 0.65;
  const mix = 1 - keep;
  return {
    ...next,
    minX: previous.minX * keep + next.minX * mix,
    maxX: previous.maxX * keep + next.maxX * mix,
    minY: previous.minY * keep + next.minY * mix,
    maxY: previous.maxY * keep + next.maxY * mix,
    width: previous.width * keep + next.width * mix,
    height: previous.height * keep + next.height * mix,
    area: previous.area * keep + next.area * mix,
    score: previous.score * keep + next.score * mix
  };
}

function maybeRequestBackendSegmentation() {
  if (state.backendSegmentationAvailable === false) return;
  if (state.backendAnalyzing || state.pinned || !state.currentTarget) return;
  if (state.frameCounter % state.backendFrameEveryN !== 0) return;
  if (!dom.workCanvas.width || !dom.workCanvas.height) return;

  state.backendAnalyzing = true;
  dom.workCanvas.toBlob(async (blob) => {
    if (!blob) {
      state.backendAnalyzing = false;
      return;
    }
    const form = new FormData();
    form.append('text', targetToPrompt(state.currentTarget));
    form.append('file', blob, 'live-frame.jpg');
    try {
      const response = await fetch(`${API_BASE}/api/search/live-frame`, {
        method: 'POST',
        body: form,
        signal: AbortSignal.timeout(5000)
      });
      if (!response.ok) throw new Error(`live frame ${response.status}`);
      const result = await response.json();
      const segmentationStatus = result.pipeline?.segmentation || '';
      const hasYolo = segmentationStatus.startsWith('yolo-segmentation:');
      state.backendSegmentationAvailable = hasYolo;
      state.backendDetections = hasYolo
        ? (result.detections || []).slice(0, 1).map((det) => normalizeServerDetection(det, result.image))
        : [];
    } catch (error) {
      console.warn('backend segmentation unavailable', error);
      state.backendSegmentationAvailable = false;
      state.backendDetections = [];
    } finally {
      state.backendAnalyzing = false;
    }
  }, 'image/jpeg', 0.78);
}

function targetToPrompt(target) {
  if (target.category === 'head') {
    return `${target.colorKey} minifigure head face printed round`;
  }
  const dimension = target.dimensionKnown === false ? '' : `${target.width}x${target.length} `;
  return `${dimension}${target.colorKey} ${target.category}`.trim();
}

function normalizeServerDetection(det, image = {}) {
  const box = det.bbox || {};
  const width = Number(box.width || 0);
  const height = Number(box.height || 0);
  return {
    ...det,
    minX: Number(box.x || 0),
    minY: Number(box.y || 0),
    maxX: Number(box.x || 0) + width,
    maxY: Number(box.y || 0) + height,
    width,
    height,
    area: Number(det.area || width * height),
    score: Number(det.score || 0),
    sourceWidth: Number(image.width || dom.workCanvas.width),
    sourceHeight: Number(image.height || dom.workCanvas.height),
    maskPolygon: det.maskPolygon || det.ransac?.polygon || null,
    source: 'backend-yolo-segmentation'
  };
}

function buildMask(imageData, colorKey) {
  const { data, width, height } = imageData;
  const mask = new Uint8Array(width * height);
  const step = 2;
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const idx = (y * width + x) * 4;
      const [h, s, l, v] = rgbToHslHsv(data[idx], data[idx + 1], data[idx + 2]);
      const matched = colorMatches(colorKey, h, s, l, v);
      if (matched) {
        for (let dy = 0; dy < step; dy += 1) {
          for (let dx = 0; dx < step; dx += 1) {
            const yy = y + dy;
            const xx = x + dx;
            if (yy < height && xx < width) mask[yy * width + xx] = 1;
          }
        }
      }
    }
  }
  dilate(mask, width, height, 1);
  return mask;
}

function colorMatches(colorKey, h, s, l, v) {
  if (colorKey === 'black') return l < 0.25 && v < 0.35;
  if (colorKey === 'white') return s < 0.22 && l > 0.72;
  if (colorKey === 'gray') return s < 0.24 && l > 0.27 && l < 0.72;
  if (s < 0.28 || l < 0.12 || l > 0.92) return false;
  if (colorKey === 'red') return h >= 342 || h <= 16;
  if (colorKey === 'blue') return h >= 190 && h <= 248;
  if (colorKey === 'yellow') return h >= 38 && h <= 68 && l > 0.28;
  if (colorKey === 'green') return h >= 82 && h <= 158;
  if (colorKey === 'orange') return h >= 15 && h <= 39;
  if (colorKey === 'purple') return h >= 255 && h <= 315;
  return false;
}

function rgbToHslHsv(r, g, b) {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === rn) h = ((gn - bn) / delta) % 6;
    else if (max === gn) h = (bn - rn) / delta + 2;
    else h = (rn - gn) / delta + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const l = (max + min) / 2;
  const s = delta === 0 ? 0 : delta / (1 - Math.abs(2 * l - 1));
  return [h, s, l, max];
}

function dilate(mask, width, height, iterations) {
  for (let iter = 0; iter < iterations; iter += 1) {
    const src = mask.slice();
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const i = y * width + x;
        if (src[i]) continue;
        if (src[i - 1] || src[i + 1] || src[i - width] || src[i + width]) mask[i] = 1;
      }
    }
  }
}

function connectedComponents(mask, width, height) {
  const visited = new Uint8Array(width * height);
  const components = [];
  const stackX = [];
  const stackY = [];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (!mask[idx] || visited[idx]) continue;
      let area = 0;
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      stackX.length = 0;
      stackY.length = 0;
      stackX.push(x);
      stackY.push(y);
      visited[idx] = 1;

      while (stackX.length) {
        const cx = stackX.pop();
        const cy = stackY.pop();
        area += 1;
        minX = Math.min(minX, cx);
        maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy);
        maxY = Math.max(maxY, cy);

        const neighbors = [
          [cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]
        ];
        for (const [nx, ny] of neighbors) {
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const ni = ny * width + nx;
          if (mask[ni] && !visited[ni]) {
            visited[ni] = 1;
            stackX.push(nx);
            stackY.push(ny);
          }
        }
      }
      components.push({ area, minX, maxX, minY, maxY });
    }
  }
  return components;
}

function drawOverlay(detections, target) {
  const canvas = dom.overlayCanvas;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  if (!target) {
    ctx.restore();
    return;
  }

  const firstDetection = detections[0] || {};
  const sourceWidth = firstDetection.sourceWidth || dom.workCanvas.width;
  const sourceHeight = firstDetection.sourceHeight || dom.workCanvas.height;
  const scale = mapVideoToCanvas(dom.cameraVideo, cssWidth, cssHeight, sourceWidth, sourceHeight);
  detections.forEach((det, index) => {
    const x = det.minX * scale.sx + scale.ox;
    const y = det.minY * scale.sy + scale.oy;
    const w = det.width * scale.sx;
    const h = det.height * scale.sy;
    const pad = Math.max(10, Math.min(w, h) * 0.16);
    const bx = x - pad;
    const by = y - pad;
    const bw = w + pad * 2;
    const bh = h + pad * 2;

    const polygon = scaledPolygon(det.maskPolygon, scale);
    if (polygon.length >= 3) drawNeonPolygon(ctx, polygon, target.colorCss);
    else drawNeonRoundRect(ctx, bx, by, bw, bh, 14, target.colorCss);

    ctx.fillStyle = 'rgba(0, 0, 0, 0.58)';
    const labelWidth = det.source === 'backend-yolo-segmentation' ? 172 : 142;
    roundRect(ctx, bx, Math.max(8, by - 34), labelWidth, 28, 10);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 13px system-ui, sans-serif';
    const label = det.source === 'backend-yolo-segmentation'
      ? `${index + 1}. ${Math.round(det.score * 100)}% mask`
      : `${index + 1}. ${Math.round(det.score * 100)}% match`;
    ctx.fillText(label, bx + 10, Math.max(27, by - 14));
  });

  if (!detections.length) {
    ctx.fillStyle = 'rgba(0,0,0,.42)';
    roundRect(ctx, cssWidth / 2 - 160, cssHeight / 2 - 34, 320, 68, 18);
    ctx.fill();
    ctx.fillStyle = '#EAEAEA';
    ctx.textAlign = 'center';
    ctx.font = '700 15px system-ui, sans-serif';
    ctx.fillText(`${target.name} 후보 탐색 중`, cssWidth / 2, cssHeight / 2 - 4);
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillStyle = '#8892B0';
    ctx.fillText('밝은 조명에서 목표 블록을 화면 중앙에 두면 더 잘 보입니다.', cssWidth / 2, cssHeight / 2 + 18);
    ctx.textAlign = 'left';
  }
  ctx.restore();
}

function scaledPolygon(polygon, scale) {
  if (!Array.isArray(polygon)) return [];
  return polygon
    .filter((point) => Array.isArray(point) && point.length >= 2)
    .map(([x, y]) => [Number(x) * scale.sx + scale.ox, Number(y) * scale.sy + scale.oy]);
}

function drawPolygonPath(ctx, polygon) {
  ctx.beginPath();
  polygon.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
}

function drawNeonPolygon(ctx, polygon, color) {
  drawNeonPath(ctx, color, () => drawPolygonPath(ctx, polygon));
}

function drawNeonRoundRect(ctx, x, y, width, height, radius, color) {
  drawNeonPath(ctx, color, () => roundRect(ctx, x, y, width, height, radius));
}

function drawNeonPath(ctx, color, buildPath) {
  const layers = [
    { width: 22, blur: 30, alpha: 0.18 },
    { width: 13, blur: 20, alpha: 0.26 },
    { width: 7, blur: 10, alpha: 0.62 },
    { width: 3, blur: 3, alpha: 1 }
  ];

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (const layer of layers) {
    buildPath();
    ctx.globalAlpha = layer.alpha;
    ctx.shadowColor = color;
    ctx.shadowBlur = layer.blur;
    ctx.lineWidth = layer.width;
    ctx.strokeStyle = color;
    ctx.stroke();
  }
  buildPath();
  ctx.globalAlpha = 0.92;
  ctx.shadowBlur = 0;
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = 'rgba(255,255,255,0.92)';
  ctx.stroke();
  ctx.restore();
}

function mapVideoToCanvas(video, canvasW, canvasH, sourceW, sourceH) {
  const videoAspect = video.videoWidth / Math.max(1, video.videoHeight);
  const canvasAspect = canvasW / Math.max(1, canvasH);
  let drawW = canvasW;
  let drawH = canvasH;
  let ox = 0;
  let oy = 0;
  if (videoAspect > canvasAspect) {
    drawH = canvasH;
    drawW = drawH * videoAspect;
    ox = (canvasW - drawW) / 2;
  } else {
    drawW = canvasW;
    drawH = drawW / videoAspect;
    oy = (canvasH - drawH) / 2;
  }
  return { sx: drawW / sourceW, sy: drawH / sourceH, ox, oy };
}

function drawMiniHeatmap(detections, target) {
  const canvas = dom.miniHeatmap;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(255,255,255,.04)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!target) return;
  const sx = canvas.width / Math.max(1, dom.workCanvas.width);
  const sy = canvas.height / Math.max(1, dom.workCanvas.height);
  detections.forEach((det) => {
    const grad = ctx.createRadialGradient(
      (det.minX + det.width / 2) * sx,
      (det.minY + det.height / 2) * sy,
      2,
      (det.minX + det.width / 2) * sx,
      (det.minY + det.height / 2) * sy,
      Math.max(det.width * sx, det.height * sy)
    );
    grad.addColorStop(0, hexToRgba(target.colorCss, 0.75));
    grad.addColorStop(1, hexToRgba(target.colorCss, 0));
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  });
}

function drawNoCameraOverlay() {
  resizeOverlay();
  const canvas = dom.overlayCanvas;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#050817';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function hexToRgba(hex, alpha) {
  const clean = hex.replace('#', '');
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function showDetail(target = state.currentTarget) {
  if (!target) return;
  state.lastScreen = document.querySelector('.screen.active')?.id || 'homeScreen';
  dom.detailName.textContent = target.name;
  dom.detailColor.textContent = target.colorLabel;
  dom.detailPartNum.textContent = target.partNum;
  dom.detailCategory.textContent = target.categoryLabel;
  dom.detailSource.textContent = target.source;
  dom.legoRender.style.setProperty('--lego-color', target.colorCss);
  showScreen('detailScreen');
}

async function loadEvalImage(file) {
  state.evalImageFile = file;
  state.evalImageBitmap = await createImageBitmap(file);
  state.evalImageNaturalSize = { width: state.evalImageBitmap.width, height: state.evalImageBitmap.height };
  dom.imageEvalFileName.textContent = file.name;
  dom.runImageEval.disabled = false;
  dom.imageEvalStatus.textContent = '이미지가 준비되었습니다. 프롬프트를 입력하고 이미지 검색을 실행하세요.';
  dom.imageEvalPlaceholder.style.display = 'none';
  dom.imageEvalSummary.textContent = '검색 대기 중';
  dom.imageEvalProposalCount.textContent = '0';
  dom.imageEvalDetectionCount.textContent = '0';
  dom.imageEvalTopScore.textContent = '0%';
  dom.imageEvalDetections.innerHTML = '';
  drawEvalBaseImage();
}

function drawEvalBaseImage() {
  const canvas = dom.imageEvalCanvas;
  const ctx = canvas.getContext('2d');
  const bitmap = state.evalImageBitmap;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#050817';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!bitmap) return null;

  const scale = Math.min(canvas.width / bitmap.width, canvas.height / bitmap.height);
  const drawW = bitmap.width * scale;
  const drawH = bitmap.height * scale;
  const offsetX = (canvas.width - drawW) / 2;
  const offsetY = (canvas.height - drawH) / 2;
  ctx.drawImage(bitmap, offsetX, offsetY, drawW, drawH);
  return { scale, offsetX, offsetY, drawW, drawH };
}

async function runImageEvaluation() {
  if (!state.evalImageFile) return;
  const query = dom.imageEvalQuery.value.trim() || dom.textQuery.value.trim() || '2x4 red brick';
  dom.runImageEval.disabled = true;
  dom.runImageEval.textContent = '검색 중...';
  dom.imageEvalStatus.textContent = `${API_BASE}/api/search/image 호출 중`;
  dom.imageEvalDetections.innerHTML = '';

  const form = new FormData();
  form.append('text', query);
  form.append('file', state.evalImageFile);
  form.append('max_results', '1');

  try {
    const res = await fetch(`${API_BASE}/api/search/image`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(30000)
    });
    if (!res.ok) throw new Error(`Backend returned ${res.status}`);
    const result = await res.json();
    renderImageEvaluationResult(result);
    dom.imageEvalStatus.textContent = '검색 완료';
  } catch (error) {
    console.error(error);
    drawEvalBaseImage();
    dom.imageEvalSummary.textContent = '검색 실패';
    dom.imageEvalStatus.textContent = `백엔드 연결 또는 이미지 처리 실패: ${error.message}`;
  } finally {
    dom.runImageEval.disabled = false;
    dom.runImageEval.textContent = '이미지 검색';
  }
}

function renderImageEvaluationResult(result) {
  const target = result.target || parseQuery(dom.imageEvalQuery.value, 'image-eval');
  const detections = result.detections || [];
  const proposals = result.proposals?.count || 0;
  const transform = drawEvalBaseImage();
  const canvas = dom.imageEvalCanvas;
  const ctx = canvas.getContext('2d');

  if (transform) {
    detections.forEach((det, index) => {
      const color = target.colorCss || COLORS[target.colorKey]?.css || '#E94560';
      const box = det.bbox;
      const x = transform.offsetX + box.x * transform.scale;
      const y = transform.offsetY + box.y * transform.scale;
      const w = box.width * transform.scale;
      const h = box.height * transform.scale;
      const pad = Math.max(5, Math.min(w, h) * 0.08);
      ctx.save();
      ctx.shadowColor = color;
      ctx.shadowBlur = index === 0 ? 22 : 10;
      ctx.lineWidth = index === 0 ? 5 : 3;
      ctx.strokeStyle = color;
      const polygon = (det.maskPolygon || det.ransac?.polygon || [])
        .filter((point) => Array.isArray(point) && point.length >= 2)
        .map(([px, py]) => [transform.offsetX + px * transform.scale, transform.offsetY + py * transform.scale]);
      if (polygon.length >= 3) drawNeonPolygon(ctx, polygon, color);
      else drawNeonRoundRect(ctx, x - pad, y - pad, w + pad * 2, h + pad * 2, 12, color);
      ctx.shadowBlur = 0;
      ctx.fillStyle = 'rgba(5, 8, 23, 0.82)';
      roundRect(ctx, x - pad, Math.max(8, y - pad - 34), 148, 28, 9);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = '700 14px system-ui, sans-serif';
      ctx.fillText(`${index + 1}. ${Math.round((det.score || 0) * 100)}%`, x - pad + 10, Math.max(27, y - pad - 15));
      ctx.restore();
    });
  }

  dom.imageEvalProposalCount.textContent = String(proposals);
  dom.imageEvalDetectionCount.textContent = String(detections.length);
  dom.imageEvalTopScore.textContent = detections.length ? `${Math.round(detections[0].score * 100)}%` : '0%';
  dom.imageEvalSummary.textContent = detections.length
    ? `${target.name || dom.imageEvalQuery.value} 후보 ${detections.length}개`
    : '후보를 찾지 못했습니다';

  dom.imageEvalDetections.innerHTML = '';
  detections.forEach((det) => {
    const item = document.createElement('div');
    item.className = 'detection-row';
    item.innerHTML = `
      <strong>#${det.rank} · ${(det.score * 100).toFixed(1)}%</strong>
      <span>bbox ${det.bbox.x}, ${det.bbox.y}, ${det.bbox.width}×${det.bbox.height}</span>
      <small>색상 ${(det.colorDominance * 100).toFixed(0)}% · 형상 ${(det.shapeScore * 100).toFixed(0)}% · 제안 ${(det.proposalScore * 100).toFixed(0)}%</small>
    `;
    dom.imageEvalDetections.appendChild(item);
  });
}

async function analyzeManualImage(file) {
  const image = await createImageBitmap(file);
  const canvas = dom.manualPreview;
  const ctx = canvas.getContext('2d');
  const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
  const w = image.width * scale;
  const h = image.height * scale;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#10182f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const colorKey = dominantColorKey(imageData);
  const target = makeTarget({ colorKey, width: 2, length: 3, category: 'brick', source: 'manual-image' });
  state.currentTarget = target;
  updateTargetUI(target);
  dom.manualPlaceholder.style.display = 'none';
  dom.analyzeManualImage.disabled = false;
}

function dominantColorKey(imageData) {
  const bins = Object.fromEntries(Object.keys(COLORS).map((key) => [key, 0]));
  const { data } = imageData;
  for (let i = 0; i < data.length; i += 16) {
    const [h, s, l, v] = rgbToHslHsv(data[i], data[i + 1], data[i + 2]);
    if (l < 0.08 || (s < 0.12 && l > 0.85)) continue;
    for (const key of Object.keys(COLORS)) {
      if (colorMatches(key, h, s, l, v)) bins[key] += key === 'black' || key === 'white' || key === 'gray' ? 0.75 : 1;
    }
  }
  return Object.entries(bins).sort((a, b) => b[1] - a[1])[0]?.[0] || 'red';
}

function renderSetParts(data) {
  dom.setName.textContent = `${data.name || data.set_num || '세트'} · ${data.set_num || ''}`;
  dom.setCount.textContent = `${data.parts?.length || 0}종`;
  dom.setPartsList.innerHTML = '';
  (data.parts || []).forEach((part) => {
    const target = normalizePart(part);
    const row = document.createElement('button');
    row.className = 'part-row';
    row.innerHTML = `
      <span class="swatch" style="background:${target.colorCss}"></span>
      <span style="flex:1;text-align:left">
        <strong>${target.colorEmoji} ${target.name} ×${target.quantity || 1}</strong>
        <small>파트 ${target.partNum} · ${target.categoryLabel}</small>
      </span>
      <span>탐색 →</span>
    `;
    row.addEventListener('click', () => startSearch(target));
    dom.setPartsList.appendChild(row);
  });
}

function normalizePart(part) {
  if (part.colorKey) return part;
  const colorKey = inferColorKey(part.color_name || part.color || part.colorLabel || 'red');
  const partName = `${part.name || part.part_name || ''} ${part.part?.name || ''}`.toLowerCase();
  let category = 'brick';
  if (partName.includes('plate') || partName.includes('플레이트')) category = 'plate';
  if (partName.includes('tile') || partName.includes('타일')) category = 'tile';
  const dimMatch = partName.match(/(\d+)\s*x\s*(\d+)/i) || [null, 2, 3];
  return makeTarget({
    colorKey,
    width: Number(dimMatch[1]),
    length: Number(dimMatch[2]),
    category,
    quantity: part.quantity || part.qty || 1,
    source: 'rebrickable-set'
  });
}

function inferColorKey(name) {
  const lower = String(name).toLowerCase();
  if (/red|빨/.test(lower)) return 'red';
  if (/blue|파/.test(lower)) return 'blue';
  if (/yellow|노/.test(lower)) return 'yellow';
  if (/green|초|녹/.test(lower)) return 'green';
  if (/orange|주황/.test(lower)) return 'orange';
  if (/purple|보라/.test(lower)) return 'purple';
  if (/black|검/.test(lower)) return 'black';
  if (/white|흰|하얀/.test(lower)) return 'white';
  if (/gray|grey|회/.test(lower)) return 'gray';
  return 'red';
}

async function loadSet() {
  const setNum = dom.setNumberInput.value.trim() || '75257-1';
  const key = dom.apiKeyInput.value.trim();
  dom.loadSetButton.disabled = true;
  dom.loadSetButton.textContent = '불러오는 중...';
  try {
    const params = new URLSearchParams();
    if (key) params.set('api_key', key);
    const res = await fetch(`${API_BASE}/api/rebrickable/set/${encodeURIComponent(setNum)}?${params.toString()}`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`Backend returned ${res.status}`);
    const data = await res.json();
    renderSetParts(data);
  } catch (error) {
    console.warn('Using sample set fallback:', error);
    renderSetParts({ ...SAMPLE_SET, set_num: setNum, name: `${setNum} 샘플/오프라인 모드` });
  } finally {
    dom.loadSetButton.disabled = false;
    dom.loadSetButton.textContent = '📦 부품 목록 불러오기';
  }
}

function setupVoice() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    dom.voiceButton.addEventListener('click', () => alert('이 브라우저는 Web Speech API를 지원하지 않습니다. 텍스트 입력을 사용해 주세요.'));
    return;
  }
  const recognition = new SpeechRecognition();
  recognition.lang = 'ko-KR';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  dom.voiceButton.addEventListener('click', () => {
    dom.voiceButton.textContent = '듣는 중...';
    recognition.start();
  });
  recognition.addEventListener('result', (event) => {
    const transcript = event.results[0][0].transcript;
    dom.textQuery.value = transcript;
    dom.voiceButton.textContent = '🎤 음성으로 말하기';
    startSearch(parseQuery(transcript, 'voice'));
  });
  recognition.addEventListener('end', () => {
    dom.voiceButton.textContent = '🎤 음성으로 말하기';
  });
}

function snapshot() {
  const canvas = document.createElement('canvas');
  canvas.width = dom.overlayCanvas.width;
  canvas.height = dom.overlayCanvas.height;
  const ctx = canvas.getContext('2d');
  try {
    ctx.drawImage(dom.cameraVideo, 0, 0, canvas.width, canvas.height);
    ctx.drawImage(dom.overlayCanvas, 0, 0);
    const link = document.createElement('a');
    link.download = `brickfinder-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  } catch (error) {
    alert('스냅샷 생성에 실패했습니다. 카메라 권한과 브라우저 보안 정책을 확인해 주세요.');
  }
}

function wireEvents() {
  $('goHome').addEventListener('click', () => showScreen('homeScreen'));
  $('openImageEval').addEventListener('click', openImageEvalScreen);
  $('openSetScreen').addEventListener('click', () => showScreen('setScreen'));
  $('openArchitecture').addEventListener('click', () => showScreen('architectureScreen'));
  $('backFromImageEval').addEventListener('click', () => showScreen('homeScreen'));
  $('backFromAr').addEventListener('click', () => showScreen('homeScreen'));
  $('backFromDetail').addEventListener('click', () => showScreen(state.lastScreen || 'homeScreen'));
  $('backFromSet').addEventListener('click', () => showScreen('homeScreen'));
  $('backFromArchitecture').addEventListener('click', () => showScreen('homeScreen'));
  $('targetDetailButton').addEventListener('click', () => showDetail());
  $('startFromDetail').addEventListener('click', () => startSearch(state.currentTarget));
  dom.startTextSearch.addEventListener('click', () => startSearch(parseQuery(dom.textQuery.value, 'text')));
  dom.textQuery.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') startSearch(parseQuery(dom.textQuery.value, 'text'));
  });
  dom.demoButton.addEventListener('click', () => {
    const target = makeTarget({ colorKey: 'blue', width: 1, length: 4, category: 'brick', source: 'demo' });
    dom.textQuery.value = '파란색 1x4 기본 브릭';
    startSearch(target);
  });
  dom.pickImageEval.addEventListener('click', () => dom.imageEvalInput.click());
  dom.imageEvalInput.addEventListener('change', async (event) => {
    const [file] = event.target.files || [];
    if (file) await loadEvalImage(file);
  });
  dom.runImageEval.addEventListener('click', runImageEvaluation);
  dom.imageEvalQuery.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && state.evalImageFile) runImageEvaluation();
  });
  dom.pickManualImage.addEventListener('click', () => dom.manualImageInput.click());
  dom.manualImageInput.addEventListener('change', async (event) => {
    const [file] = event.target.files || [];
    if (file) await analyzeManualImage(file);
  });
  dom.analyzeManualImage.addEventListener('click', () => startSearch(parseQuery(dom.textQuery.value, 'live-camera')));
  dom.pinButton.addEventListener('click', () => {
    state.pinned = !state.pinned;
    dom.pinButton.textContent = state.pinned ? '📌 핀 해제' : '📌 핀 고정';
    if (state.pinned) dom.cameraVideo.pause();
    else dom.cameraVideo.play();
  });
  dom.snapshotButton.addEventListener('click', snapshot);
  $('loadSetButton').addEventListener('click', loadSet);
  window.addEventListener('resize', resizeOverlay);
  window.addEventListener('hashchange', () => {
    if (location.hash === '#image-eval') openImageEvalScreen();
  });
}

function init() {
  wireEvents();
  setupVoice();
  renderRecent();
  const initialTarget = parseQuery(dom.textQuery.value, 'text');
  state.currentTarget = initialTarget;
  updateTargetUI(initialTarget);
  renderSetParts(SAMPLE_SET);
  if (location.hash === '#image-eval') {
    openImageEvalScreen();
  } else {
    startSearch(initialTarget);
  }
}

init();

import * as THREE from 'three/webgpu';
import {
  Fn, uniform, float, vec3, instancedArray, instanceIndex, uv,
  positionGeometry, positionWorld, sin, cos, pow, smoothstep, mix,
  sqrt, select, hash, time, deltaTime, PI, mx_noise_float,
  pass, mrt, output, transformedNormalView,
} from 'three/tsl';
import { dof } from 'three/addons/tsl/display/DepthOfFieldNode.js';

// Config: Enable fog and DoF effects (true = better visuals with black background, false = transparent canvas showing distortion grid)
const USE_FOG_AND_DOF = true;

// ── All tunable parameters ──────────────────────────────────────
const CONFIG = {
  // Grid & field
  BLADE_COUNT:       16129,    // total blades (must be GRID_DIM²)
  GRID_DIM:          127,      // grid resolution per axis
  FIELD_SIZE:        28,       // world units the field spans
  FIELD_OFFSET_X:    0,        // horizontal field shift

  // Camera
  CAMERA_FOV:        38,
  CAMERA_POS:        [3, 7, 18] as [number, number, number],
  CAMERA_TARGET:     [3, 1.5, 0] as [number, number, number],

  // Renderer
  MAX_DPR:           1.0,      // cap pixel ratio
  TONE_EXPOSURE:     1.0,

  // Scene
  FOG_DENSITY:       0.035,    // FogExp2 density
  FOG_COLOR:         '#000000',
  BG_COLOR:          '#000000',

  // Mouse interaction
  MOUSE_RADIUS:      6.1,
  MOUSE_STRENGTH:    4.0,
  OUTER_RADIUS:      9.4,
  OUTER_STRENGTH:    1.45,

  // Camera sphere push (static camera — pre-computed)
  CAM_SPHERE_RADIUS: 15.0,
  CAM_SPHERE_STRENGTH_BASE: 5.9,

  // Wind
  WIND_SPEED:        1.3,
  WIND_AMPLITUDE:    0.21,
  WIND_FREQ_X1:      0.35,     // primary wind wave X frequency
  WIND_FREQ_Z1:      0.12,     // primary wind wave Z frequency
  WIND_FREQ_X2:      0.18,     // secondary wind wave X frequency
  WIND_FREQ_Z2:      0.28,     // secondary wind wave Z frequency
  WIND_PHASE:        1.7,      // secondary wave phase offset
  WIND_SPEED_RATIO:  0.67,     // secondary wind speed multiplier
  WIND_Z_RATIO:      0.55,     // Z amplitude ratio relative to X
  WIND_LERP_SPEED:   4.0,      // how fast wind builds up

  // Blade shape
  BLADE_WIDTH:       4.0,
  BLADE_TIP_WIDTH:   0.19,
  BLADE_HEIGHT:      1.6,
  BLADE_HEIGHT_VAR:  0.5,
  BLADE_HEIGHT_BASE: 0.35,     // minimum height offset
  BLADE_LEAN:        1.1,
  BLADE_BEND_EXP:    1.8,      // bend curve exponent

  // Blade geometry (mesh)
  BLADE_SEGMENTS:    4,
  BLADE_MESH_WIDTH:  0.055,
  BLADE_MESH_HEIGHT: 1.0,
  BLADE_TAPER:       0.82,     // width reduction from base to tip

  // Noise / clumping
  NOISE_AMPLITUDE:   1.85,
  NOISE_FREQUENCY:   0.3,
  NOISE2_AMPLITUDE:  0.2,
  NOISE2_FREQUENCY:  15,

  // Boundary / culling
  BOUNDARY_RADIUS:   8,       // base radius for blade culling
  BOUNDARY_NOISE_MUL:6.0,      // edge noise multiplier
  BOUNDARY_FALLOFF:  1.0,      // smoothstep inner offset
  BOUNDARY_THRESHOLD:0.05,     // below this = fully culled

  // Colors
  BLADE_BASE_COLOR:  '#0e1e04',
  BLADE_TIP_COLOR:   '#c8b840',
  MID_COLOR:         '#2d4e0e',
  GOLDEN_TIP_COLOR:  '#d4b838',
  GREEN_TIP_COLOR:   '#4a7a14',
  GROUND_COLOR:      '#000000',
  BLADE_COLOR_VAR:   0.93,

  // Ground
  GROUND_RADIUS:     14.0,
  GROUND_FALLOFF:    2.4,
  GROUND_NOISE_FREQ: 0.25,
  GROUND_NOISE_OFF:  100,

  // Fog
  FOG_START:         10.0,
  FOG_END:           20.0,
  FOG_INTENSITY:     1.0,
  FOG_COLOR:         '#000000',

  // DoF
  DOF_FOCUS_DIST:    22.0,
  DOF_FOCAL_LENGTH:  10.0,
  DOF_BOKEH_SCALE:   12.5,

  // Push response
  PUSH_LERP_FAST:    12.0,     // when push increases
  PUSH_LERP_SLOW:    1.0,      // when push decreases
  DISTANCE_EPSILON:  0.0001,
};

export async function initGrass() {
  const grassCanvas = document.getElementById('grass-canvas') as HTMLCanvasElement | null;
  if (!grassCanvas) return;

  const BLADE_COUNT = CONFIG.BLADE_COUNT;
  const FIELD_SIZE = CONFIG.FIELD_SIZE;
  const GRID_DIM = CONFIG.GRID_DIM;
  const FIELD_OFFSET_X = CONFIG.FIELD_OFFSET_X;

  // Scene setup (transparent or solid based on USE_FOG_AND_DOF)
  const scene = new THREE.Scene();
  if (USE_FOG_AND_DOF) {
    scene.background = new THREE.Color('#000000');
    scene.fog = new THREE.FogExp2(CONFIG.FOG_COLOR, CONFIG.FOG_DENSITY);
  } else {
    scene.background = null;
    // No fog — transparency mode for showing distortion grid
  }

  // Camera
  const camera = new THREE.PerspectiveCamera(CONFIG.CAMERA_FOV, innerWidth / innerHeight, 0.1, 100);
  camera.position.set(...CONFIG.CAMERA_POS);
  camera.lookAt(...CONFIG.CAMERA_TARGET);

  // Renderer (alpha when not using fog/DoF so distortion shows through)
  const renderer = new THREE.WebGPURenderer({ canvas: grassCanvas, antialias: false, alpha: !USE_FOG_AND_DOF, powerPreference: 'low-power' });
  const maxDPR = Math.min(devicePixelRatio, CONFIG.MAX_DPR);
  renderer.setPixelRatio(maxDPR);
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = CONFIG.TONE_EXPOSURE;
  if (!USE_FOG_AND_DOF) renderer.setClearColor(0x000000, 0);
  await renderer.init();

  // GPU Buffers
  const bladeData = instancedArray(BLADE_COUNT, 'vec4');
  const bendState = instancedArray(BLADE_COUNT, 'vec4');
  const bladeBound = instancedArray(BLADE_COUNT, 'float');

  // Uniforms
  const mouseWorld = uniform(new THREE.Vector3(99999, 0, 99999));
  const mouseRadius = uniform(CONFIG.MOUSE_RADIUS);
  const mouseStrength = uniform(CONFIG.MOUSE_STRENGTH);
  const outerRadius = uniform(CONFIG.OUTER_RADIUS);
  const outerStrength = uniform(CONFIG.OUTER_STRENGTH);
  const camSphereWorld = uniform(new THREE.Vector3(99999, 0, 99999));
  const camSphereRadius = uniform(CONFIG.CAM_SPHERE_RADIUS);
  const camSphereStrength = uniform(CONFIG.CAM_SPHERE_STRENGTH_BASE);
  const grassDensity = uniform(1.0);
  const windSpeed = uniform(CONFIG.WIND_SPEED);
  const windAmplitude = uniform(CONFIG.WIND_AMPLITUDE);
  const bladeWidth = uniform(CONFIG.BLADE_WIDTH);
  const bladeTipWidth = uniform(CONFIG.BLADE_TIP_WIDTH);
  const bladeHeight = uniform(CONFIG.BLADE_HEIGHT);
  const bladeHeightVariation = uniform(CONFIG.BLADE_HEIGHT_VAR);
  const bladeLean = uniform(CONFIG.BLADE_LEAN);
  const noiseAmplitude = uniform(CONFIG.NOISE_AMPLITUDE);
  const noiseFrequency = uniform(CONFIG.NOISE_FREQUENCY);
  const noise2Amplitude = uniform(CONFIG.NOISE2_AMPLITUDE);
  const noise2Frequency = uniform(CONFIG.NOISE2_FREQUENCY);
  const bladeColorVariation = uniform(CONFIG.BLADE_COLOR_VAR);
  const groundRadius = uniform(CONFIG.GROUND_RADIUS);
  const groundFalloff = uniform(CONFIG.GROUND_FALLOFF);
  const bladeBaseColor = uniform(new THREE.Color(CONFIG.BLADE_BASE_COLOR));
  const bladeTipColor = uniform(new THREE.Color(CONFIG.BLADE_TIP_COLOR));
  const groundColor = uniform(new THREE.Color(CONFIG.GROUND_COLOR));
  const fogStart = uniform(CONFIG.FOG_START);
  const fogEnd = uniform(CONFIG.FOG_END);
  const fogIntensity = uniform(CONFIG.FOG_INTENSITY);
  const fogColor = uniform(new THREE.Color(CONFIG.FOG_COLOR));
  const goldenTipColor = uniform(new THREE.Color(CONFIG.GOLDEN_TIP_COLOR));
  const greenTipColor = uniform(new THREE.Color(CONFIG.GREEN_TIP_COLOR));
  const midColor = uniform(new THREE.Color(CONFIG.MID_COLOR));

  // DoF Uniforms
  const focusDistanceU = uniform(CONFIG.DOF_FOCUS_DIST);
  const focalLengthU = uniform(CONFIG.DOF_FOCAL_LENGTH);
  const bokehScaleU = uniform(CONFIG.DOF_BOKEH_SCALE);

  // Shader-only uniforms (not exposed to GPU compute, used in TSL nodes)
  const boundaryRadiusU = uniform(CONFIG.BOUNDARY_RADIUS);
  const boundaryNoiseMulU = uniform(CONFIG.BOUNDARY_NOISE_MUL);
  const boundaryFalloffU = uniform(CONFIG.BOUNDARY_FALLOFF);
  const boundaryThresholdU = uniform(CONFIG.BOUNDARY_THRESHOLD);
  const windFreqX1U = uniform(CONFIG.WIND_FREQ_X1);
  const windFreqZ1U = uniform(CONFIG.WIND_FREQ_Z1);
  const windFreqX2U = uniform(CONFIG.WIND_FREQ_X2);
  const windFreqZ2U = uniform(CONFIG.WIND_FREQ_Z2);
  const windPhaseU = uniform(CONFIG.WIND_PHASE);
  const windSpeedRatioU = uniform(CONFIG.WIND_SPEED_RATIO);
  const windZRatioU = uniform(CONFIG.WIND_Z_RATIO);
  const windLerpSpeedU = uniform(CONFIG.WIND_LERP_SPEED);
  const pushLerpFastU = uniform(CONFIG.PUSH_LERP_FAST);
  const pushLerpSlowU = uniform(CONFIG.PUSH_LERP_SLOW);
  const distEpsilonU = uniform(CONFIG.DISTANCE_EPSILON);
  const bladeHeightBaseU = uniform(CONFIG.BLADE_HEIGHT_BASE);
  const bladeBendExpU = uniform(CONFIG.BLADE_BEND_EXP);
  const grassDensityThresholdU = uniform(0.5);
  const groundNoiseFreqU = uniform(CONFIG.GROUND_NOISE_FREQ);
  const groundNoiseOffU = uniform(CONFIG.GROUND_NOISE_OFF);
  const noiseSeedU = uniform(50);
  const noiseSeedBoundaryU = uniform(100);

  // Color gradient uniforms
  const colorLowerGradStartU = uniform(0.0);
  const colorLowerGradEndU = uniform(0.45);
  const colorUpperGradStartU = uniform(0.4);
  const colorUpperGradEndU = uniform(0.85);
  const goldenRatioU = uniform(0.4);
  const opacityFadeDistU = uniform(2.0);
  const opacityFadeStartU = uniform(5.0);
  const opacityStartU = uniform(0.0);
  const opacityEndU = uniform(0.1);

  // Noise
  const noise2D = Fn(([x, z]) => mx_noise_float(vec3(x, float(0), z)).mul(0.5).add(0.5));

  // ── Compute Init ──────────────────────────────────────────────
  const computeInit = Fn(() => {
    const blade = bladeData.element(instanceIndex);
    const col = instanceIndex.mod(GRID_DIM);
    const row = instanceIndex.div(GRID_DIM);
    const jx = hash(instanceIndex).sub(0.5);
    const jz = hash(instanceIndex.add(7919)).sub(0.5);
    const wx = col.toFloat().add(jx).div(float(GRID_DIM)).sub(0.5).mul(FIELD_SIZE).add(FIELD_OFFSET_X);
    const wz = row.toFloat().add(jz).div(float(GRID_DIM)).sub(0.5).mul(FIELD_SIZE);
    blade.x.assign(wx);
    blade.y.assign(wz);
    blade.z.assign(hash(instanceIndex.add(1337)).mul(PI.mul(2)));
    const n1 = noise2D(wx.mul(noiseFrequency), wz.mul(noiseFrequency));
    const n2 = noise2D(wx.mul(noiseFrequency.mul(noise2Frequency)).add(noiseSeedU), wz.mul(noiseFrequency.mul(noise2Frequency)).add(noiseSeedU));
    const clump = n1.mul(noiseAmplitude).sub(noise2Amplitude).add(n2.mul(noise2Amplitude).mul(2)).max(0);
    blade.w.assign(clump);
    const dist = sqrt(wx.mul(wx).add(wz.mul(wz)));
    const edgeNoise = noise2D(wx.mul(groundNoiseFreqU).add(noiseSeedBoundaryU), wz.mul(groundNoiseFreqU).add(noiseSeedBoundaryU));
    const maxR = boundaryRadiusU.add(edgeNoise.sub(0.5).mul(boundaryNoiseMulU));
    const boundary = float(1).sub(smoothstep(maxR.sub(boundaryFalloffU), maxR, dist));
    bladeBound.element(instanceIndex).assign(select(boundary.lessThan(boundaryThresholdU), float(0), boundary));
  })().compute(BLADE_COUNT);

  // ── Compute Update ───────────────────────────────────────────
  const computeUpdate = Fn(() => {
    const blade = bladeData.element(instanceIndex);
    const bend = bendState.element(instanceIndex);
    const bx = blade.x;
    const bz = blade.y;

    const w1 = sin(bx.mul(windFreqX1U).add(bz.mul(windFreqZ1U)).add(time.mul(windSpeed)));
    const w2 = sin(bx.mul(windFreqX2U).add(bz.mul(windFreqZ2U)).add(time.mul(windSpeed.mul(windSpeedRatioU))).add(windPhaseU));
    const windX = w1.add(w2).mul(windAmplitude);
    const windZ = w1.sub(w2).mul(windAmplitude.mul(windZRatioU));

    const lw = deltaTime.mul(windLerpSpeedU).saturate();
    bend.x.assign(mix(bend.x, windX, lw));
    bend.y.assign(mix(bend.y, windZ, lw));

    // Mouse push
    const dx = bx.sub(mouseWorld.x);
    const dz = bz.sub(mouseWorld.z);
    const dist = sqrt(dx.mul(dx).add(dz.mul(dz))).add(distEpsilonU);
    const falloff = float(1).sub(dist.div(mouseRadius).saturate());
    const influence = falloff.mul(falloff).mul(mouseStrength);
    const pushX = dx.div(dist).mul(influence);
    const pushZ = dz.div(dist).mul(influence);

    // Outer mouse sphere
    const odx = bx.sub(mouseWorld.x);
    const odz = bz.sub(mouseWorld.z);
    const odist = sqrt(odx.mul(odx).add(odz.mul(odz))).add(distEpsilonU);
    const ofalloff = float(1).sub(odist.div(outerRadius).saturate());
    const oinfluence = ofalloff.mul(ofalloff).mul(outerStrength);
    const opushX = odx.div(odist).mul(oinfluence);
    const opushZ = odz.div(odist).mul(oinfluence);

    // Camera sphere push
    const cdx = bx.sub(camSphereWorld.x);
    const cdz = bz.sub(camSphereWorld.z);
    const cdist = sqrt(cdx.mul(cdx).add(cdz.mul(cdz))).add(distEpsilonU);
    const cfalloff = float(1).sub(cdist.div(camSphereRadius).saturate());
    const cinfluence = cfalloff.mul(cfalloff).mul(camSphereStrength);
    const cpushX = cdx.div(cdist).mul(cinfluence);
    const cpushZ = cdz.div(cdist).mul(cinfluence);

    const totalPushX = pushX.add(opushX).add(cpushX);
    const totalPushZ = pushZ.add(opushZ).add(cpushZ);

    const targetMag = sqrt(totalPushX.mul(totalPushX).add(totalPushZ.mul(totalPushZ)));
    const currentMag = sqrt(bend.z.mul(bend.z).add(bend.w.mul(bend.w)));
    const lm = select(targetMag.greaterThan(currentMag), deltaTime.mul(pushLerpFastU), deltaTime.mul(pushLerpSlowU)).saturate();
    bend.z.assign(mix(bend.z, totalPushX, lm));
    bend.w.assign(mix(bend.w, totalPushZ, lm));
  })().compute(BLADE_COUNT);

  // ── Blade Geometry ────────────────────────────────────────────
  function createBladeGeometry() {
    const segs = CONFIG.BLADE_SEGMENTS, W = CONFIG.BLADE_MESH_WIDTH, H = CONFIG.BLADE_MESH_HEIGHT;
    const verts: number[] = [], norms: number[] = [], uvArr: number[] = [], idx: number[] = [];
    for (let i = 0; i <= segs; i++) {
      const t = i / segs, y = t * H, hw = W * 0.5 * (1.0 - t * CONFIG.BLADE_TAPER);
      verts.push(-hw, y, 0, hw, y, 0);
      norms.push(0, 0, 1, 0, 0, 1);
      uvArr.push(0, t, 1, t);
    }
    for (let i = 0; i < segs; i++) { const b = i * 2; idx.push(b, b + 1, b + 2, b + 1, b + 3, b + 2); }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(norms, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvArr, 2));
    geo.setIndex(idx);
    return geo;
  }

  // ── Grass Material ────────────────────────────────────────────
  const grassMat = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide, fog: USE_FOG_AND_DOF });

  grassMat.positionNode = Fn(() => {
    const blade = bladeData.element(instanceIndex);
    const bend = bendState.element(instanceIndex);
    const worldX = blade.x, worldZ = blade.y, rotY = blade.z;
    const boundary = bladeBound.element(instanceIndex);
    const visible = select(hash(instanceIndex.add(9999)).lessThan(grassDensity.mul(grassDensityThresholdU)), float(1), float(0));
    const hVar = hash(instanceIndex.add(5555)).mul(bladeHeightVariation);
    const heightScale = bladeHeightBaseU.add(blade.w).add(hVar).mul(boundary).mul(visible);
    const taper = float(1).sub(uv().y.mul(float(1).sub(bladeTipWidth)));
    const lx = positionGeometry.x.mul(bladeWidth).mul(taper).mul(heightScale.sign());
    const ly = positionGeometry.y.mul(heightScale).mul(bladeHeight);
    const cY = cos(rotY), sY = sin(rotY);
    const rx = lx.mul(cY), rz = lx.mul(sY);
    const t = uv().y;
    const bendFactor = pow(t, bladeBendExpU);
    const staticBendX = hash(instanceIndex.add(7777)).sub(0.5).mul(bladeLean);
    const staticBendZ = hash(instanceIndex.add(8888)).sub(0.5).mul(bladeLean);
    const bendX = staticBendX.add(bend.x).add(bend.z);
    const bendZ = staticBendZ.add(bend.y).add(bend.w);
    const relX = rx.add(bendX.mul(bendFactor).mul(bladeHeight));
    const relY = ly;
    const relZ = rz.add(bendZ.mul(bendFactor).mul(bladeHeight));
    const origLen = sqrt(rx.mul(rx).add(ly.mul(ly)).add(rz.mul(rz)));
    const newLen = sqrt(relX.mul(relX).add(relY.mul(relY)).add(relZ.mul(relZ)));
    const scale = origLen.div(newLen.max(distEpsilonU));
    return vec3(worldX.add(relX.mul(scale)), relY.mul(scale), worldZ.add(relZ.mul(scale)));
  })();

  grassMat.colorNode = USE_FOG_AND_DOF
    ? Fn(() => {
        const t = uv().y;
        const clump = bladeData.element(instanceIndex).w.saturate();
        const bladeHash = hash(instanceIndex.add(4242));
        const isGolden = bladeHash.lessThan(goldenRatioU);
        const lowerGrad = smoothstep(colorLowerGradStartU, colorLowerGradEndU, t);
        const upperGrad = smoothstep(colorUpperGradStartU, colorUpperGradEndU, t);
        const tipMix = float(1).sub(bladeColorVariation).add(clump.mul(bladeColorVariation));
        const greenTip = mix(greenTipColor, bladeTipColor, tipMix);
        const warmTip = mix(greenTipColor, goldenTipColor, tipMix);
        const tipFinal = mix(greenTip, warmTip, select(isGolden, float(1), float(0)));
        const lowerColor = mix(bladeBaseColor, midColor, lowerGrad);
        const grassColor = mix(lowerColor, tipFinal, upperGrad);
        const blade = bladeData.element(instanceIndex);
        const dist = sqrt(blade.x.mul(blade.x).add(blade.y.mul(blade.y)));
        const fogFactor = smoothstep(fogStart, fogEnd, dist).mul(fogIntensity);
        return mix(grassColor, fogColor, fogFactor);
      })()
    : Fn(() => {
        const t = uv().y;
        const clump = bladeData.element(instanceIndex).w.saturate();
        const bladeHash = hash(instanceIndex.add(4242));
        const isGolden = bladeHash.lessThan(goldenRatioU);
        const lowerGrad = smoothstep(colorLowerGradStartU, colorLowerGradEndU, t);
        const upperGrad = smoothstep(colorUpperGradStartU, colorUpperGradEndU, t);
        const tipMix = float(1).sub(bladeColorVariation).add(clump.mul(bladeColorVariation));
        const greenTip = mix(greenTipColor, bladeTipColor, tipMix);
        const warmTip = mix(greenTipColor, goldenTipColor, tipMix);
        const tipFinal = mix(greenTip, warmTip, select(isGolden, float(1), float(0)));
        const lowerColor = mix(bladeBaseColor, midColor, lowerGrad);
        const grassColor = mix(lowerColor, tipFinal, upperGrad);
        return grassColor;
      })();

  grassMat.opacityNode = Fn(() => {
    const blade = bladeData.element(instanceIndex);
    const dist = sqrt(blade.x.mul(blade.x).add(blade.y.mul(blade.y)));
    const fadeEnd = select(fogIntensity.greaterThan(0.01), fogEnd.add(opacityFadeDistU), float(15.0));
    const fadeFactor = float(1).sub(smoothstep(fadeEnd.sub(opacityFadeStartU), fadeEnd, dist));
    return smoothstep(opacityStartU, opacityEndU, uv().y).mul(fadeFactor);
  })();
  grassMat.transparent = true;

  // ── Instances ────────────────────────────────────────────────
  const bladeGeo = createBladeGeometry();
  const grass = new THREE.InstancedMesh(bladeGeo, grassMat, BLADE_COUNT);
  grass.frustumCulled = true;
  scene.add(grass);
  const dummy = new THREE.Object3D();
  for (let i = 0; i < BLADE_COUNT; i++) grass.setMatrixAt(i, dummy.matrix);
  grass.instanceMatrix.needsUpdate = true;

  // ── Ground ────────────────────────────────────────────────────
  const groundMat = new THREE.MeshBasicNodeMaterial();
  if (USE_FOG_AND_DOF) {
    groundMat.colorNode = Fn(() => {
      const wx = positionWorld.x, wz = positionWorld.z;
      const dist = sqrt(wx.mul(wx).add(wz.mul(wz)));
      const edgeNoise = noise2D(wx.mul(groundNoiseFreqU).add(groundNoiseOffU), wz.mul(groundNoiseFreqU).add(groundNoiseOffU));
      const maxR = groundRadius.add(edgeNoise.sub(0.5).mul(boundaryNoiseMulU));
      const t = smoothstep(maxR.sub(groundFalloff), maxR, dist);
      return mix(groundColor, fogColor, t);
    })();
  } else {
    groundMat.colorNode = Fn(() => {
      const wx = positionWorld.x, wz = positionWorld.z;
      const dist = sqrt(wx.mul(wx).add(wz.mul(wz)));
      const edgeNoise = noise2D(wx.mul(groundNoiseFreqU).add(groundNoiseOffU), wz.mul(groundNoiseFreqU).add(groundNoiseOffU));
      const maxR = groundRadius.add(edgeNoise.sub(0.5).mul(boundaryNoiseMulU));
      const t = smoothstep(maxR.sub(groundFalloff), maxR, dist);
      return mix(groundColor, groundColor, t);
    })();
    groundMat.opacityNode = Fn(() => {
      const wx = positionWorld.x, wz = positionWorld.z;
      const dist = sqrt(wx.mul(wx).add(wz.mul(wz)));
      const edgeNoise = noise2D(wx.mul(groundNoiseFreqU).add(groundNoiseOffU), wz.mul(groundNoiseFreqU).add(groundNoiseOffU));
      const maxR = groundRadius.add(edgeNoise.sub(0.5).mul(boundaryNoiseMulU));
      const t = smoothstep(maxR.sub(groundFalloff), maxR, dist);
      return float(1).sub(t);
    })();
    groundMat.transparent = true;
  }
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(FIELD_SIZE * 3, FIELD_SIZE * 3), groundMat);
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  // ── Lighting ─────────────────────────────────────────────────
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xfff4e0, 1.5);
  dirLight.position.set(5, 10, 7);
  scene.add(dirLight);

  // ── Post Processing (DoF when USE_FOG_AND_DOF is true) ───────
  const postProcessing = new THREE.PostProcessing(renderer);
  const scenePass = pass(scene, camera);
  scenePass.setMRT(mrt({
    output: output,
    normal: transformedNormalView,
  }));
  const sceneColor = scenePass.getTextureNode('output');

  if (USE_FOG_AND_DOF) {
    const sceneViewZ = scenePass.getViewZNode();
    const dofOutput = dof(sceneColor, sceneViewZ, focusDistanceU, focalLengthU, bokehScaleU);
    postProcessing.outputNode = dofOutput;
  } else {
    postProcessing.outputNode = sceneColor;
  }
  postProcessing.needsUpdate = true;

  // ── Mouse (throttled to rAF cadence) ──────────────────────────
  const raycaster = new THREE.Raycaster();
  const mouseNDC = new THREE.Vector2();
  const grassPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const hitPoint = new THREE.Vector3();
  let mouseDirty = false;

  window.addEventListener('mousemove', (e) => {
    mouseNDC.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    mouseDirty = true;
  });
  window.addEventListener('mouseleave', () => {
    mouseWorld.value.set(99999, 0, 99999);
    mouseDirty = false;
  });

  // ── Resize ────────────────────────────────────────────────────
  let resizeTimeout: ReturnType<typeof setTimeout>;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      if (window.innerWidth < 768) return;
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      const dpr = Math.min(devicePixelRatio, 1.0);
      renderer.setPixelRatio(dpr);
      renderer.setSize(innerWidth, innerHeight);
    }, 100);
  });

  // ── Boot ──────────────────────────────────────────────────────
  renderer.compute(computeInit);

  // Pre-warm — render a few frames before fading in
  async function prewarm() {
    for (let i = 0; i < 3; i++) {
      renderer.compute(computeUpdate);
      postProcessing.render();
      await new Promise(r => requestAnimationFrame(r));
    }
    grassCanvas.classList.add('visible');
  }
  prewarm();

  // ── Animate ───────────────────────────────────────────────────
  const clock = new THREE.Clock();

  // 40fps cap for CPU optimization
  let lastGrassFrame = 0;
  const GRASS_FPS = 40;
  const GRASS_FRAME_INTERVAL = 1000 / GRASS_FPS;

  // Camera sphere push — pre-computed (camera is static)
  const camHeight = camera.position.y;
  const proximityT = Math.max(0, 1 - camHeight / 10);
  const proxCurve = proximityT * proximityT;
  camSphereWorld.value.set(camera.position.x, 0, camera.position.z);
  camSphereRadius.value = Math.min(15, 15 * (0.3 + proxCurve * 0.7));
  camSphereStrength.value = 5.9 * (0.1 + proxCurve * 0.9);

  function animate(timestamp: number) {
    // 40fps throttle
    if (timestamp - lastGrassFrame < GRASS_FRAME_INTERVAL) return;
    lastGrassFrame = timestamp;

    // Throttled mouse raycasting — only when mouse actually moved
    if (mouseDirty) {
      raycaster.setFromCamera(mouseNDC, camera);
      if (raycaster.ray.intersectPlane(grassPlane, hitPoint)) {
        mouseWorld.value.copy(hitPoint);
      }
      mouseDirty = false;
    }

    renderer.compute(computeUpdate);
    postProcessing.render();
  }

  renderer.setAnimationLoop(animate);
}

if (window.innerWidth >= 768) {
  await initGrass();
} else {
  const c = document.getElementById('grass-canvas');
  if (c) c.style.display = 'none';
}

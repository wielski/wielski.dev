// Mouse glow follower
const glow = document.getElementById('glow') as HTMLElement | null;
let mx = -500, my = -500;

document.addEventListener('mousemove', e => {
  mx = e.clientX;
  my = e.clientY;
  if (glow) {
    glow.style.left = mx + 'px';
    glow.style.top = my + 'px';
  }
});

// 3D Distortion Canvas
const canvas = document.getElementById('distortion') as HTMLCanvasElement | null;
const ctx = canvas?.getContext('2d');
let w = 0, h = 0;
let cols = 0, rows = 0;
const spacing = 55;
let points: { x: number; y: number; ox: number; oy: number; vx: number; vy: number }[] = [];
let animFrame: number;

function resize() {
  if (!canvas || !ctx) return;
  w = canvas.width = window.innerWidth;
  h = canvas.height = window.innerHeight;
  cols = Math.ceil(w / spacing) + 2;
  rows = Math.ceil(h / spacing) + 2;
  initPoints();
}

function initPoints() {
  points = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      points.push({
        x: c * spacing,
        y: r * spacing,
        ox: c * spacing,
        oy: r * spacing,
        vx: 0,
        vy: 0
      });
    }
  }
}

function dist(x1: number, y1: number, x2: number, y2: number) {
  return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

let time = 0;

// Frame rate limiting for CPU optimization
let lastDistortionFrame = 0;
const TARGET_DISTORTION_FPS = 30;
const DISTORTION_FRAME_INTERVAL = 1000 / TARGET_DISTORTION_FPS;

function dr(timestamp: number) {
  if (!ctx || !canvas) return;
  if (timestamp - lastDistortionFrame < DISTORTION_FRAME_INTERVAL) {
    animFrame = requestAnimationFrame(dr);
    return;
  }
  lastDistortionFrame = timestamp;

  ctx.clearRect(0, 0, w, h);
  time += 0.008;

  const scrollY = window.scrollY;

  for (const p of points) {
    const d = dist(mx, my + scrollY, p.ox, p.oy);
    const maxDist = 200;
    const influence = Math.max(0, 1 - d / maxDist);
    const angle = Math.atan2(p.oy - (my + scrollY), p.ox - mx);
    const push = influence * 25;

    const waveX = Math.sin(p.oy * 0.008 + time) * 3;
    const waveY = Math.cos(p.ox * 0.008 + time * 0.7) * 3;

    const targetX = p.ox + Math.cos(angle) * push + waveX;
    const targetY = p.oy + Math.sin(angle) * push + waveY;

    p.vx += (targetX - p.x) * 0.08;
    p.vy += (targetY - p.y) * 0.08;
    p.vx *= 0.85;
    p.vy *= 0.85;
    p.x += p.vx;
    p.y += p.vy;
  }

  ctx.strokeStyle = 'rgba(255,255,255,0.025)';
  ctx.lineWidth = 0.5;

  // horizontal lines
  for (let r = 0; r < rows; r++) {
    ctx.beginPath();
    for (let c = 0; c < cols; c++) {
      const p = points[r * cols + c];
      const py = p.y - scrollY;
      if (c === 0) ctx.moveTo(p.x, py);
      else ctx.lineTo(p.x, py);
    }
    ctx.stroke();
  }

  // vertical lines
  for (let c = 0; c < cols; c++) {
    ctx.beginPath();
    for (let r = 0; r < rows; r++) {
      const p = points[r * cols + c];
      const py = p.y - scrollY;
      if (r === 0) ctx.moveTo(p.x, py);
      else ctx.lineTo(p.x, py);
    }
    ctx.stroke();
  }

  // intersection dots near cursor
  for (const p of points) {
    const d = dist(mx, my + scrollY, p.x, p.y);
    if (d < 150) {
      const alpha = (1 - d / 150) * 0.15;
      ctx.fillStyle = `rgba(255,255,255,${alpha})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y - scrollY, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  animFrame = requestAnimationFrame(dr);
}

window.addEventListener('resize', resize);
resize();
dr();

// Bubble tilt on hover
document.querySelectorAll('.bubble').forEach(bubble => {
  bubble.addEventListener('mousemove', e => {
    const rect = bubble.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    bubble.style.transform = `perspective(600px) rotateY(${x * 4}deg) rotateX(${-y * 4}deg)`;
  });
  bubble.addEventListener('mouseleave', () => {
    bubble.style.transform = 'perspective(600px) rotateY(0deg) rotateX(0deg)';
    bubble.style.transition = 'transform 0.4s ease';
    setTimeout(() => bubble.style.transition = '', 400);
  });
});

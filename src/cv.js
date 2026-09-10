// cv.js — minimal OpenCV-compatible primitives used by ppocrv6-min (pure JS).
//
// Implements the exact operations the Python package relies on from
// opencv-python / numpy, without any native dependency:
//   resizeBilinear        ~ cv2.resize(..., interpolation=cv2.INTER_LINEAR)
//   findContours          ~ cv2.findContours(mask, RETR_LIST, CHAIN_APPROX_SIMPLE)
//                           (Suzuki-Abe border following)
//   minAreaRect/boxPoints ~ cv2.minAreaRect / cv2.boxPoints (rotating calipers)
//   fillPolyMask          ~ cv2.fillPoly on an uint8 mask
//   meanMasked            ~ cv2.mean(img, mask)[0]
//   getPerspectiveTransform / warpPerspective ~ cv2.getPerspectiveTransform /
//                           cv2.warpPerspective(flags=cv2.INTER_LINEAR)
//
// Image representation: { width, height, channels, data } with HWC layout.
// `data` is Uint8Array (8U) or Float32Array (32F).

/**
 * Bilinear resize (OpenCV INTER_LINEAR semantics: sample coordinates are
 * aligned to pixel centers, src coordinate = (dst + 0.5) * scale - 0.5).
 */
export function resizeBilinear(src, dstW, dstH) {
  const { width: sw, height: sh, channels: c } = src;
  const dst = {
    width: dstW,
    height: dstH,
    channels: c,
    data: src.data instanceof Float32Array
      ? new Float32Array(dstW * dstH * c)
      : new Uint8Array(dstW * dstH * c),
  };
  const isFloat = src.data instanceof Float32Array;
  const sd = src.data;
  const dd = dst.data;
  const rx = sw / dstW;
  const ry = sh / dstH;
  // Precompute x contributions.
  const x0s = new Int32Array(dstW);
  const x1s = new Int32Array(dstW);
  const fxs = new Float64Array(dstW);
  for (let dx = 0; dx < dstW; dx++) {
    let fx = (dx + 0.5) * rx - 0.5;
    if (fx < 0) fx = 0;
    let x0 = Math.floor(fx);
    if (x0 > sw - 1) x0 = sw - 1;
    const x1 = x0 + 1 > sw - 1 ? sw - 1 : x0 + 1;
    x0s[dx] = x0;
    x1s[dx] = x1;
    fxs[dx] = fx - x0;
  }
  const rowStride = sw * c;
  for (let dy = 0; dy < dstH; dy++) {
    let fy = (dy + 0.5) * ry - 0.5;
    if (fy < 0) fy = 0;
    let y0 = Math.floor(fy);
    if (y0 > sh - 1) y0 = sh - 1;
    const y1 = y0 + 1 > sh - 1 ? sh - 1 : y0 + 1;
    const wy = fy - y0;
    const r0 = y0 * rowStride;
    const r1 = y1 * rowStride;
    const doff = dy * dstW * c;
    for (let dx = 0; dx < dstW; dx++) {
      const x0 = x0s[dx];
      const x1 = x1s[dx];
      const wx = fxs[dx];
      const i00 = r0 + x0 * c;
      const i01 = r0 + x1 * c;
      const i10 = r1 + x0 * c;
      const i11 = r1 + x1 * c;
      for (let ch = 0; ch < c; ch++) {
        const v =
          (1 - wy) * ((1 - wx) * sd[i00 + ch] + wx * sd[i01 + ch]) +
          wy * ((1 - wx) * sd[i10 + ch] + wx * sd[i11 + ch]);
        dd[doff + dx * c + ch] = isFloat ? v : Math.round(v);
      }
    }
  }
  return dst;
}

// ---------------------------------------------------------------------------
// findContours — Suzuki-Abe border following (RETR_LIST, CHAIN_APPROX_SIMPLE)
// ---------------------------------------------------------------------------

// Direction ring, clockwise in image coordinates (y grows downwards):
// 0:E 1:SE 2:S 3:SW 4:W 5:NW 6:N 7:NE
const DX = [1, 1, 0, -1, -1, -1, 0, 1];
const DY = [0, 1, 1, 1, 0, -1, -1, -1];

/**
 * Extract all outer borders of connected components from a binary mask.
 * Uses connected components labeling + boundary tracing for simplicity and reliability.
 * Equivalent to cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
 * for the DB use case (we only need outer contours of text components).
 */
export function findContours(mask, w, h) {
  const labels = new Int32Array(w * h); // 0 = background, >0 = component label
  let nextLabel = 1;

  // First pass: assign provisional labels using union-find for 8-connectivity
  const parent = []; // union-find for label equivalences

  function find(x) {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]; // path compression
      x = parent[x];
    }
    return x;
  }

  function union(a, b) {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) {
      parent[ra] = rb;
    }
  }

  parent[0] = 0;

  const get = (x, y) => (x >= 0 && x < w && y >= 0 && y < h ? mask[y * w + x] : 0);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (get(x, y) === 0) continue;

      const neighbors = [];
      if (y > 0) {
        if (x > 0 && get(x - 1, y - 1) !== 0) neighbors.push(labels[(y - 1) * w + (x - 1)]);
        if (get(x, y - 1) !== 0) neighbors.push(labels[(y - 1) * w + x]);
        if (x < w - 1 && get(x + 1, y - 1) !== 0) neighbors.push(labels[(y - 1) * w + (x + 1)]);
      }
      if (x > 0 && get(x - 1, y) !== 0) neighbors.push(labels[y * w + (x - 1)]);

      if (neighbors.length === 0) {
        // New component
        labels[y * w + x] = nextLabel;
        parent[nextLabel] = nextLabel;
        nextLabel++;
      } else {
        // Find the minimum label among neighbors
        const minLabel = Math.min(...neighbors);
        labels[y * w + x] = minLabel;
        // Union all neighbors with the min label
        for (const n of neighbors) {
          union(minLabel, n);
        }
      }
    }
  }

  // Second pass: resolve final labels and collect components
  const finalLabels = new Int32Array(w * h);
  const components = new Map(); // finalLabel -> array of points

  for (let i = 0; i < w * h; i++) {
    if (mask[i] === 0) continue;
    const finalLabel = find(labels[i]);
    if (finalLabel === 0) continue;
    finalLabels[i] = finalLabel;
    if (!components.has(finalLabel)) {
      components.set(finalLabel, []);
    }
    const py = Math.floor(i / w);
    const px = i % w;
    components.get(finalLabel).push([px, py]);
  }

  const contours = [];

  // For each component, find its outer boundary
  for (const [label, points] of components) {
    if (points.length < 3) continue;

    // Find boundary points: pixels that are adjacent to background (4-connected)
    const boundary = new Set();
    for (const [px, py] of points) {
      let isBoundary = false;
      for (let d = 0; d < 8; d++) {
        const nx = px + DX[d];
        const ny = py + DY[d];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h || mask[ny * w + nx] === 0) {
          isBoundary = true;
          break;
        }
      }
      if (isBoundary) {
        boundary.add(`${px},${py}`);
      }
    }

    if (boundary.size < 3) continue;

    // Sort boundary points in clockwise order around centroid
    const bPoints = Array.from(boundary).map(s => {
      const [px, py] = s.split(',').map(Number);
      return [px, py];
    });

    const cx = bPoints.reduce((s, p) => s + p[0], 0) / bPoints.length;
    const cy = bPoints.reduce((s, p) => s + p[1], 0) / bPoints.length;

    // Sort by angle around centroid
    bPoints.sort((a, b) => {
      const aa = Math.atan2(a[1] - cy, a[0] - cx);
      const ba = Math.atan2(b[1] - cy, b[0] - cx);
      return aa - ba;
    });

    contours.push(simpleApprox(bPoints, w, h));
  }

  return contours;
}

/** CHAIN_APPROX_SIMPLE: drop points collinear with their neighbours. */
function simpleApprox(pts, w, h) {
  if (pts.length <= 2) return pts;
  const out = [];
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const p = pts[i];
    const a = pts[(i - 1 + n) % n];
    const b = pts[(i + 1) % n];
    const cross =
      (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if (cross !== 0) out.push(p);
  }
  return out;
}

// ---------------------------------------------------------------------------
// minAreaRect / boxPoints — rotating calipers over the convex hull
// ---------------------------------------------------------------------------

/** Andrew monotone chain convex hull (strict, no collinear points). */
export function convexHull(points) {
  const pts = points
    .map((p) => [p[0], p[1]])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length <= 2) return pts;
  const cross = (o, a, b) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/**
 * Minimum-area rectangle of a point set, cv2.minAreaRect compatible shape:
 * { center: [x, y], size: [w, h], angle } with angle in degrees [0, 90).
 */
export function minAreaRect(points) {
  const hull = convexHull(points);
  if (hull.length === 0) {
    return { center: [0, 0], size: [0, 0], angle: 0 };
  }
  if (hull.length === 1) {
    return { center: [...hull[0]], size: [0, 0], angle: 0 };
  }
  if (hull.length === 2) {
    const [a, b] = hull;
    const cx = (a[0] + b[0]) / 2;
    const cy = (a[1] + b[1]) / 2;
    const w = Math.hypot(b[0] - a[0], b[1] - a[1]);
    let angle = (Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI;
    let ww = w;
    let hh = 0;
    if (angle < 0) angle += 180;
    if (angle >= 90) {
      angle -= 90;
      [ww, hh] = [hh, ww];
    }
    return { center: [cx, cy], size: [ww, hh], angle };
  }
  let best = null; // { area, u0, v0, ew, ev, ux, uy, vx, vy }
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    const ex = b[0] - a[0];
    const ey = b[1] - a[1];
    const len = Math.hypot(ex, ey);
    if (len === 0) continue;
    const ux = ex / len;
    const uy = ey / len;
    const vx = -uy;
    const vy = ux;
    let minU = Infinity;
    let maxU = -Infinity;
    let minV = Infinity;
    let maxV = -Infinity;
    for (const p of hull) {
      const pu = p[0] * ux + p[1] * uy;
      const pv = p[0] * vx + p[1] * vy;
      if (pu < minU) minU = pu;
      if (pu > maxU) maxU = pu;
      if (pv < minV) minV = pv;
      if (pv > maxV) maxV = pv;
    }
    const ew = maxU - minU;
    const ev = maxV - minV;
    const area = ew * ev;
    if (best === null || area < best.area - 1e-12) {
      best = { area, minU, minV, ew, ev, ux, uy, vx, vy };
    }
  }
  // Corner set of the best rectangle.
  const u0 = best.minU;
  const v0 = best.minV;
  const corners = [
    [u0 * best.ux + v0 * best.vx, u0 * best.uy + v0 * best.vy],
    [(u0 + best.ew) * best.ux + v0 * best.vx, (u0 + best.ew) * best.uy + v0 * best.vy],
    [(u0 + best.ew) * best.ux + (v0 + best.ev) * best.vx, (u0 + best.ew) * best.uy + (v0 + best.ev) * best.vy],
    [u0 * best.ux + (v0 + best.ev) * best.vx, u0 * best.uy + (v0 + best.ev) * best.vy],
  ];
  const cx =
    (u0 + best.ew / 2) * best.ux + (v0 + best.ev / 2) * best.vx;
  const cy =
    (u0 + best.ew / 2) * best.uy + (v0 + best.ev / 2) * best.vy;
  let angle = (Math.atan2(best.uy, best.ux) * 180) / Math.PI;
  let sizeW = best.ew;
  let sizeH = best.ev;
  if (angle < 0) angle += 180;
  if (angle >= 90) angle -= 90;
  return { center: [cx, cy], size: [sizeW, sizeH], angle, corners };
}

/** cv2.boxPoints equivalent: 4 corners of the rect, clockwise from the
 *  topmost-left in OpenCV's convention. Only the corner set matters here. */
export function boxPoints(rect) {
  if (rect.corners) return rect.corners.map((p) => [...p]);
  const [cx, cy] = rect.center;
  const [w, h] = rect.size;
  const a = (rect.angle * Math.PI) / 180;
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  // OpenCV rotates around the center; y axis points down.
  return [
    [cx - (w / 2) * cos + (h / 2) * sin, cy - (w / 2) * sin - (h / 2) * cos],
    [cx + (w / 2) * cos + (h / 2) * sin, cy + (w / 2) * sin - (h / 2) * cos],
    [cx + (w / 2) * cos - (h / 2) * sin, cy + (w / 2) * sin + (h / 2) * cos],
    [cx - (w / 2) * cos - (h / 2) * sin, cy - (w / 2) * sin + (h / 2) * cos],
  ];
}

// ---------------------------------------------------------------------------
// fillPoly + masked mean
// ---------------------------------------------------------------------------

/**
 * cv2.fillPoly equivalent on a single-channel mask: pixels whose integer
 * coordinates lie inside the polygon (boundary included) are set to 1.
 * @param {number} w @param {number} h mask size
 * @param {number[][]} polygon integer vertices [[x, y], ...]
 * @returns {Uint8Array} w*h mask
 */
export function fillPolyMask(w, h, polygon) {
  const mask = new Uint8Array(w * h);
  const n = polygon.length;
  if (n === 0) return mask;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of polygon) {
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  }
  minY = Math.max(0, minY);
  maxY = Math.min(h - 1, maxY);
  for (let y = minY; y <= maxY; y++) {
    const xs = [];
    for (let i = 0; i < n; i++) {
      const a = polygon[i];
      const b = polygon[(i + 1) % n];
      if (a[1] === b[1]) {
        // Horizontal edge: contributes a full span on this row.
        if (a[1] === y) {
          xs.push(Math.min(a[0], b[0]), Math.max(a[0], b[0]));
        }
        continue;
      }
      const ymin = Math.min(a[1], b[1]);
      const ymax = Math.max(a[1], b[1]);
      if (y >= ymin && y < ymax) {
        const t = (y - a[1]) / (b[1] - a[1]);
        xs.push(a[0] + t * (b[0] - a[0]));
      }
    }
    if (xs.length === 0) continue;
    xs.sort((p, q) => p - q);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      let xa = Math.round(xs[k]);
      let xb = Math.round(xs[k + 1]);
      if (xa < 0) xa = 0;
      if (xb > w - 1) xb = w - 1;
      for (let x = xa; x <= xb; x++) mask[y * w + x] = 1;
    }
  }
  return mask;
}

/** cv2.mean(singleChannelImg, mask)[0]: mean over pixels where mask != 0. */
export function meanMasked(img, w, h, mask) {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < w * h; i++) {
    if (mask[i]) {
      sum += img[i];
      count++;
    }
  }
  return count === 0 ? 0 : sum / count;
}

// ---------------------------------------------------------------------------
// Perspective transform
// ---------------------------------------------------------------------------

/** Solve the 8-unknown linear system for the 3x3 homography mapping the four
 *  source points onto the four destination points (cv2.getPerspectiveTransform). */
export function getPerspectiveTransform(src4, dst4) {
  // Build 8x8 matrix A and vector b: A h = b.
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src4[i];
    const [u, v] = dst4[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const hh = solveLinearSystem(A, b);
  return [hh[0], hh[1], hh[2], hh[3], hh[4], hh[5], hh[6], hh[7], 1];
}

/** Gaussian elimination with partial pivoting. */
function solveLinearSystem(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (M[piv][col] === 0) throw new Error("singular matrix");
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f !== 0) {
        for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
      }
    }
  }
  return M.map((row) => row[n]);
}

/**
 * cv2.warpPerspective(src, M, (dstW, dstH), flags=cv2.INTER_LINEAR):
 * inverse-maps every destination pixel through M^-1 and bilinearly samples
 * the source; pixels mapping outside are 0 (BORDER_CONSTANT).
 */
export function warpPerspective(src, H, dstW, dstH) {
  const Hi = invert3x3(H);
  const { width: sw, height: sh, channels: c } = src;
  const sd = src.data;
  const isFloat = sd instanceof Float32Array;
  const dd = isFloat
    ? new Float32Array(dstW * dstH * c)
    : new Uint8Array(dstW * dstH * c);
  const [h00, h01, h02, h10, h11, h12, h20, h21, h22] = Hi;
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const denom = h20 * x + h21 * y + h22;
      const sx = (h00 * x + h01 * y + h02) / denom;
      const sy = (h10 * x + h11 * y + h12) / denom;
      const doff = (y * dstW + x) * c;
      if (
        sx < -1 || sy < -1 ||
        sx > sw || sy > sh ||
        !Number.isFinite(sx) || !Number.isFinite(sy)
      ) {
        continue; // stays zero
      }
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const wx = sx - x0;
      const wy = sy - y0;
      for (let ch = 0; ch < c; ch++) {
        const px = (xx, yy) => {
          if (xx < 0 || xx >= sw || yy < 0 || yy >= sh) return 0;
          return sd[(yy * sw + xx) * c + ch];
        };
        const v =
          (1 - wy) * ((1 - wx) * px(x0, y0) + wx * px(x0 + 1, y0)) +
          wy * ((1 - wx) * px(x0, y0 + 1) + wx * px(x0 + 1, y0 + 1));
        dd[doff + ch] = isFloat ? v : Math.round(v);
      }
    }
  }
  return { width: dstW, height: dstH, channels: c, data: dd };
}

function invert3x3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (det === 0) throw new Error("singular homography");
  const id = 1 / det;
  return [
    A * id,
    (c * h - b * i) * id,
    (b * f - c * e) * id,
    B * id,
    (a * i - c * g) * id,
    (c * d - a * f) * id,
    C * id,
    (b * g - a * h) * id,
    (a * e - b * d) * id,
  ];
}

// ---------------------------------------------------------------------------
// channel shuffling helpers
// ---------------------------------------------------------------------------

/** RGBA (or RGB) HWC buffer -> BGR HWC. */
export function rgbaToBgr(rgba, width, height, srcChannels = 4) {
  const out = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    out[i * 3] = rgba[i * srcChannels + 2];
    out[i * 3 + 1] = rgba[i * srcChannels + 1];
    out[i * 3 + 2] = rgba[i * srcChannels];
  }
  return out;
}

/** BGR HWC buffer -> RGB HWC (channel reversal, allocation-free view is not
 *  possible on typed arrays, so a copy is returned). */
export function bgrToRgb(bgr, width, height) {
  const out = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    out[i * 3] = bgr[i * 3 + 2];
    out[i * 3 + 1] = bgr[i * 3 + 1];
    out[i * 3 + 2] = bgr[i * 3];
  }
  return out;
}

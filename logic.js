// Pure rules shared by index.html and test.mjs. Same rules as the v1 Python (organize.py + pipeline.py).
export const TINY = 480;          // longest side (px) below this = low-res
export const NEAR = 6;            // max pHash distance for a near-duplicate
export const FACE_SCORE = 0.5;    // face detector confidence that counts as a person
export const FACE_MIN = 0.012;    // smaller faces (share of the photo's long side) don't count
export const BLUR = 10;           // Laplacian variance of the largest face below this = blurry. Sample: blurred 6,
                                  // motion-blurred 6, soft night shot 15, sharp ones 57+
export const EDITED = ['-edited'];  // Takeout's suffix for her edited copy (localized in some languages: add it here)
export const ALBUMS_DIR = 'אלבומים';
// every image type Takeout may hold; ones the browser can't open (HEIC, TIFF) are counted as unreadable, not skipped
export const IMAGES = ['.jpg', '.jpeg', '.jfif', '.png', '.webp', '.avif', '.gif', '.bmp', '.heic', '.heif', '.tif', '.tiff'];

const pad = n => String(n).padStart(2, '0');
export const fmt = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
  `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;  // local time

// Takeout's .json for image `name` among the .json names in its folder, or null. Handles IMG.jpg.json /
// IMG.jpg.supplemental-metadata.json, names cut to 46 chars, IMG(1).jpg <-> IMG.jpg(1).json, IMG-edited.jpg.
export function sidecar(name, jsons) {
  const m = /^(.*?)(\(\d+\))?(\.\w+)$/.exec(name);
  if (!m) return null;
  let [, stem, n = '', ext] = m;
  for (const s of EDITED) if (stem.endsWith(s)) stem = stem.slice(0, -s.length);
  const full = stem + ext + '.supplemental-metadata', need = Math.min((stem + ext).length, 46);
  let best = null;
  for (const j of jsons) {
    if (!j.endsWith('.json')) continue;
    let base = j.slice(0, -5);
    if (n) {
      if (!base.endsWith(n)) continue;
      base = base.slice(0, -n.length);
    }
    if (full.startsWith(base) && base.length >= need && (!best || base.length > best[0])) best = [base.length, j];
  }
  return best && best[1];
}

export function takenFromJson(text) {
  try {
    const ts = parseInt(JSON.parse(text).photoTakenTime.timestamp, 10);
    return ts > 0 ? fmt(new Date(ts * 1000)) : null;
  } catch {
    return null;
  }
}

const splitExt = p => {
  const i = p.lastIndexOf('.');
  return i > p.lastIndexOf('/') ? [p.slice(0, i), p.slice(i)] : [p, ''];
};

// IMG.jpg + IMG-edited.jpg in the same folder -> Map {original: edited}
export function editedOriginals(paths) {
  const have = new Set(paths), out = new Map();
  for (const p of paths) {
    const [stem, ext] = splitExt(p);
    for (const s of EDITED) if (stem.endsWith(s) && have.has(stem.slice(0, -s.length) + ext)) out.set(stem.slice(0, -s.length) + ext, p);
  }
  return out;
}

export function rejectReason(w, h, faces, faceSharp) {
  if (Math.max(w, h) < TINY) return 'low-res';
  if (!faces) return 'no people';
  if (faceSharp != null && faceSharp < BLUR) return 'blurry';
  return null;
}

// Variance of the 4-neighbour Laplacian (low = blurry), on a w*h gray array.
export function lapVar(g, w, h) {
  let n = 0, sum = 0, sq = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, v = g[i - w] + g[i + w] + g[i - 1] + g[i + 1] - 4 * g[i];
    n++; sum += v; sq += v * v;
  }
  return n ? sq / n - (sum / n) ** 2 : 0;
}

// pHash of a 32x32 gray image (row-major): 8x8 lowest DCT frequencies, each bit = above their median. 16 hex chars.
const DCT = Array.from({length: 8}, (_, u) => Float64Array.from({length: 32}, (_, x) => Math.cos(Math.PI * u * (2 * x + 1) / 64)));
export function phash(g) {
  const tmp = DCT.map(cu => Float64Array.from({length: 32}, (_, c) => cu.reduce((s, k, r) => s + k * g[r * 32 + c], 0)));
  const low = tmp.flatMap(t => DCT.map(cv => cv.reduce((s, k, c) => s + k * t[c], 0)));
  const s = [...low].sort((a, b) => a - b), med = (s[31] + s[32]) / 2;
  return BigInt('0b' + low.map(v => (v > med ? '1' : '0')).join('')).toString(16).padStart(16, '0');
}

const pop = x => {
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
};
const halves = h => [parseInt(h.slice(0, 8), 16), parseInt(h.slice(8), 16)];
export const hamming = (a, b) => {
  const [a1, a2] = halves(a), [b1, b2] = halves(b);
  return pop(a1 ^ b1) + pop(a2 ^ b2);
};

// items [{id, phash, pixels, size}] -> Map dup_id -> {keeper, dist}. Group = pairs within maxdist; keeper = most
// pixels, then biggest file. Only members within maxdist of the keeper are dropped (chains keep the far end).
export function nearDups(items, maxdist) {
  const n = items.length, hi = new Uint32Array(n), lo = new Uint32Array(n), parent = Int32Array.from({length: n}, (_, i) => i);
  items.forEach((it, i) => [hi[i], lo[i]] = halves(it.phash));
  const find = x => {
    while (parent[x] !== x) x = parent[x] = parent[parent[x]];
    return x;
  };
  // ponytail: O(n^2), ~1 s for 10k photos; bucket by hash prefix if libraries get much bigger
  for (let i = 0; i < n - 1; i++) for (let j = i + 1; j < n; j++) {
    if (pop(hi[i] ^ hi[j]) + pop(lo[i] ^ lo[j]) <= maxdist) parent[find(j)] = find(i);
  }
  const groups = new Map();
  for (let i = 0; i < n; i++) groups.set(find(i), [...(groups.get(find(i)) || []), i]);
  const out = new Map();
  for (const g of groups.values()) {
    if (g.length < 2) continue;
    const k = g.reduce((a, b) => ((items[b].pixels || 0) > (items[a].pixels || 0) ||
      ((items[b].pixels || 0) === (items[a].pixels || 0) && (items[b].size || 0) > (items[a].size || 0)) ? b : a));
    for (const i of g) {
      const d = pop(hi[i] ^ hi[k]) + pop(lo[i] ^ lo[k]);
      if (i !== k && d <= maxdist) out.set(items[i].id, {keeper: items[k].id, dist: d});
    }
  }
  return out;
}

// photo records -> Map path -> {reason, dupOf, dist}; reason null = kept. In order, at most one reason each:
// unreadable, duplicate (an -edited copy replaces its original), low-res / no people / blurry, similar.
// Paths in `keep` (rescued on the review) skip everything after duplicates.
export function decide(photos, keep) {
  const out = new Map(photos.map(p => [p.path, {reason: null, dupOf: null, dist: null}]));
  const set = (p, reason, dupOf = null, dist = null) => out.set(p, {reason, dupOf, dist});
  const ok = photos.filter(p => !p.err);
  photos.filter(p => p.err).forEach(p => set(p.path, 'unreadable'));
  const byPath = new Map(ok.map(p => [p.path, p])), replaced = new Map();
  for (const [orig, ed] of editedOriginals(ok.map(p => p.path))) replaced.set(byPath.get(orig).hash, ed);
  for (const p of ok) if (replaced.has(p.hash) && replaced.get(p.hash) !== p.path) set(p.path, 'duplicate', replaced.get(p.hash));
  const keeper = new Map(), isEdited = p => EDITED.some(s => splitExt(p)[0].endsWith(s));
  const rank = p => [isEdited(p.path) ? 0 : 1, p.path.length, p.path];
  for (const p of ok.filter(p => !replaced.has(p.hash)).sort((a, b) => {
    const [x, y] = [rank(a), rank(b)];
    return x[0] - y[0] || x[1] - y[1] || (x[2] < y[2] ? -1 : x[2] > y[2] ? 1 : 0);
  })) {
    if (!keeper.has(p.hash)) keeper.set(p.hash, p.path);
    else set(p.path, 'duplicate', keeper.get(p.hash));
  }
  const free = () => ok.filter(p => out.get(p.path).reason === null && !keep.has(p.path));
  for (const p of free()) {
    const r = rejectReason(p.w, p.h, p.faces, p.faceSharp);
    if (r) set(p.path, r);
  }
  const items = free().filter(p => p.phash).map(p => ({id: p.path, phash: p.phash, pixels: p.w * p.h, size: p.size}));
  for (const [id, {keeper: k, dist}] of nearDups(items, NEAR)) set(id, 'similar', k, dist);
  return out;
}

export const safeName = s => s.replace(/[\\/:*?"<>|]/g, '-').replace(/[. ]+$/, '');

// Kept photos (in sorting order) -> one copy per chosen album: {path, folder, name}. Names are date-prefixed so
// albums sort by date; a clash in one album (Windows ignores case) gets " (2)", " (3)", ...
export function albumFiles(photos, albumNames) {
  const used = new Set(), out = [];
  for (const p of photos) {
    if (!Array.isArray(p.choice)) continue;
    const base = p.path.split('/').pop(), [stem, ext] = splitExt((p.taken ? p.taken.slice(0, 10) + '_' : '') + base);
    for (const a of p.choice) {
      const folder = safeName(albumNames[a]);
      let name = stem + ext;
      for (let k = 2; used.has((folder + '/' + name).toLowerCase()); k++) name = `${stem} (${k})${ext}`;
      used.add((folder + '/' + name).toLowerCase());
      out.push({path: p.path, folder, name});
    }
  }
  return out;
}

// node grandma/web/test.mjs — prints ok
process.env.TZ = 'Asia/Jerusalem';
import assert from 'node:assert/strict';
import * as L from './logic.js';

// --- Takeout sidecars and dates (same quirks as v1) ---
const J = ['IMG_1.jpg.json', 'IMG_1.jpg(1).json', 'IMG_2.jpg.supplemental-metadata.json',
  'PXL_20230101_123456789.MP.jpg.supplemental-met.json', 'IMG_3.jpg.supplemental-metadata(2).json'];
assert.equal(L.sidecar('IMG_1.jpg', J), 'IMG_1.jpg.json');
assert.equal(L.sidecar('IMG_1(1).jpg', J), 'IMG_1.jpg(1).json');
assert.equal(L.sidecar('IMG_1-edited.jpg', J), 'IMG_1.jpg.json');
assert.equal(L.sidecar('IMG_2.jpg', J), 'IMG_2.jpg.supplemental-metadata.json');
assert.equal(L.sidecar('PXL_20230101_123456789.MP.jpg', J), 'PXL_20230101_123456789.MP.jpg.supplemental-met.json');
assert.equal(L.sidecar('IMG_3(2).jpg', J), 'IMG_3.jpg.supplemental-metadata(2).json');
assert.equal(L.sidecar('IMG_4.jpg', J), null);
assert.equal(L.sidecar('IMG_1(2).jpg', J), null);  // another photo's sidecar is never borrowed
assert.equal(L.sidecar('a'.repeat(47) + '.jpg', ['a'.repeat(46) + '.json']), 'a'.repeat(46) + '.json');
assert.equal(L.takenFromJson('{"photoTakenTime": {"timestamp": "1262347200"}}'), '2010-01-01 14:00:00');  // local time
assert.equal(L.takenFromJson('not json'), null);
assert.equal(L.takenFromJson('{"photoTakenTime": {"timestamp": "0"}}'), null);

// --- quality rules ---
assert.deepEqual([...L.editedOriginals(['x/IMG_1.jpg', 'x/IMG_1-edited.jpg', 'y/IMG_1.jpg', 'x/IMG_2.jpg'])],
  [['x/IMG_1.jpg', 'x/IMG_1-edited.jpg']]);
assert.equal(L.rejectReason(479, 300, 2, 99), 'low-res');
assert.equal(L.rejectReason(480, 300, 2, 99), null);
assert.equal(L.rejectReason(4000, 3000, 0, null), 'no people');
assert.equal(L.rejectReason(4000, 3000, 1, L.BLUR - 0.1), 'blurry');
assert.equal(L.rejectReason(4000, 3000, 1, L.BLUR), null);
const flat = new Float32Array(100).fill(7), checker = Float32Array.from({length: 100}, (_, i) => ((i % 10) + Math.floor(i / 10)) % 2 * 255);
assert.equal(L.lapVar(flat, 10, 10), 0);
assert.ok(L.lapVar(checker, 10, 10) > 1000);

// --- pHash: bit-identical to the old pipeline.analyze on the same 32x32 gray ---
const img = f => Float64Array.from({length: 1024}, (_, i) => f(i % 32, Math.floor(i / 32)));
const wave = (x, y) => 128 + 100 * Math.sin(x / 5) * Math.cos(y / 7);
const blob = (x, y) => 128 + 90 * Math.exp(-((x - 10) ** 2 + (y - 12) ** 2) / 40) - 60 * Math.exp(-((x - 22) ** 2 + (y - 20) ** 2) / 60) + 2 * x;
const noise = (x, y) => (x * 7 + y * 13) % 5 - 2;
assert.equal(L.phash(img(wave)), 'b54a4ab54ab54ab5');  // both values computed by the Python version
assert.equal(L.phash(img(blob)), '82873938c7c7c739');
const a = img(blob), noisy = img((x, y) => blob(x, y) + noise(x, y)), other = img((x, y) => (x * y) % 255);
assert.equal(L.hamming(L.phash(a), L.phash(a)), 0);
assert.ok(L.hamming(L.phash(a), L.phash(noisy)) <= 6);
assert.ok(L.hamming(L.phash(a), L.phash(other)) > 6);

// --- decide: same order and cases as v1's clean() ---
const FAR = ['0000000000000000', 'ffffffffffffffff', '00000000ffffffff', 'ffffffff00000000',
  '0f0f0f0f0f0f0f0f', 'f0f0f0f0f0f0f0f0', '3333333333333333'];
const P = (path, hash, phash, o = {}) => ({path, hash, phash, w: 4000, h: 3000, size: 100, faces: 1, faceSharp: 99, ...o});
const photos = [
  P('x/d1.jpg', 'm1', FAR[0]), P('y/d1.jpg', 'm1', FAR[0]),  // exact copies
  P('x/IMG_1.jpg', 'm2', FAR[1]), P('x/IMG_1-edited.jpg', 'm3', FAR[2]),
  P('x/keep.jpg', 'm4', FAR[3], {faces: 0}), P('x/land.jpg', 'm5', FAR[4], {faces: 0}),
  P('x/s0.jpg', 'm6', FAR[5]), P('x/s1.jpg', 'm7', FAR[5].slice(0, -1) + '1', {w: 3000}),  // smaller: similar
  P('x/bad.jpg', 'm8', null, {err: 'DOMException: decode', w: null, h: null}),
  P('x/q0.jpg', 'm9', FAR[6], {size: 900}), P('x/q1.jpg', 'm10', FAR[6].slice(0, -1) + '1', {size: 300}),  // bigger file wins
  P('x/סבא וסבתא 1960.jpg', 'm11', '5555555555555555', {faces: 0}),  // rescued; name has spaces
  P('x/n0.jpg', 'm12', '9999999999999999', {faces: 0}), P('x/n1.jpg', 'm13', '9999999999999998', {w: 3000}),  // twin survives
];
const got = Object.fromEntries([...L.decide(photos, new Set(['x/keep.jpg', 'x/סבא וסבתא 1960.jpg']))]
  .map(([p, d]) => [p, [d.reason, d.dupOf]]));
assert.deepEqual(got, {
  'x/d1.jpg': [null, null], 'y/d1.jpg': ['duplicate', 'x/d1.jpg'],
  'x/IMG_1.jpg': ['duplicate', 'x/IMG_1-edited.jpg'], 'x/IMG_1-edited.jpg': [null, null],
  'x/keep.jpg': [null, null], 'x/land.jpg': ['no people', null],
  'x/s0.jpg': [null, null], 'x/s1.jpg': ['similar', 'x/s0.jpg'], 'x/bad.jpg': ['unreadable', null],
  'x/q0.jpg': [null, null], 'x/q1.jpg': ['similar', 'x/q0.jpg'],
  'x/סבא וסבתא 1960.jpg': [null, null], 'x/n0.jpg': ['no people', null], 'x/n1.jpg': [null, null],
});

// --- album files ---
assert.deepEqual(L.albumFiles([
  {path: 'a/IMG_1.jpg', taken: '2001-02-03 10:00:00', choice: [0]},
  {path: 'b/IMG_1.jpg', taken: '2001-02-03 11:00:00', choice: [0, 2]},  // same date + name in album 0
  {path: 'c/x.png', taken: null, choice: [3]},
  {path: 'd/y.jpg', taken: '2002-01-01 00:00:00', choice: 'drop'},
  {path: 'e/z.jpg', taken: null},  // undecided
], ['A', 'B/1', 'C', 'D']), [
  {path: 'a/IMG_1.jpg', folder: 'A', name: '2001-02-03_IMG_1.jpg'},
  {path: 'b/IMG_1.jpg', folder: 'A', name: '2001-02-03_IMG_1 (2).jpg'},
  {path: 'b/IMG_1.jpg', folder: 'C', name: '2001-02-03_IMG_1.jpg'},
  {path: 'c/x.png', folder: 'D', name: 'x.png'},
]);
assert.deepEqual(L.albumFiles([{path: 'a/x.jpg', choice: [0]}, {path: 'b/X.JPG', choice: [0]}], ['A']).map(f => f.name),
  ['x.jpg', 'X (2).JPG']);  // Windows file names ignore case
assert.equal(L.safeName('B/1: "x"?'), 'B-1- -x--');  // Windows-safe folder names
for (const e of ['.bmp', '.avif', '.jfif', '.tif', '.tiff']) assert.ok(L.IMAGES.includes(e), e);  // old scans: never skipped silently

console.log('ok');

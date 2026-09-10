import argparse
import json
import os
import sys
import time
import numpy as np
import cv2
import onnxruntime as ort

# Path resolution
_LOCAL_ROOT = r"D:\models\PP-OCRv6"
_env = os.environ.get("PPOCR_DET_DIR")
_local = os.path.join(_LOCAL_ROOT, "PP-OCRv6_tiny_det_onnx")
if _env and os.path.isfile(os.path.join(_env, "inference.onnx")):
    det_dir = _env
else:
    det_dir = _local

_env = os.environ.get("PPOCR_REC_DIR")
_local = os.path.join(_LOCAL_ROOT, "PP-OCRv6_tiny_rec_onnx")
if _env and os.path.isfile(os.path.join(_env, "inference.onnx")):
    rec_dir = _env
else:
    rec_dir = _local

# Det: DB + resize + normalize
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def imread_unicode(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def _get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    i1, i2, i3, i4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        i1, i4 = 0, 1
    else:
        i1, i4 = 1, 0
    if points[3][1] > points[2][1]:
        i2, i3 = 2, 3
    else:
        i2, i3 = 3, 2
    return [points[i1], points[i2], points[i3], points[i4]], min(bounding_box[1])

def _unclip(box, ratio):
    box = np.asarray(box, dtype=np.float64)
    x = box[:, 0]
    y = box[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    closed = np.vstack([box, box[:1]])
    length = float(np.linalg.norm(np.diff(closed, axis=1), axis=1).sum())
    distance = area * ratio / (length + 1e-9)
    import pyclipper
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return offset.Execute(distance)

def db_postprocess(pred, dest_w, dest_h, thresh=0.2, box_thresh=0.4, unclip_ratio=1.4, min_size=3):
    mask = (pred > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = pred.shape
    boxes, scores = [], []
    for contour in contours[:3000]:
        points, sside = _get_mini_boxes(contour)
        if sside < min_size:
            continue
        points = np.array(points)
        # score
        box = points.astype(np.float32).copy()
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        m2 = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(m2, box.reshape(1, -1, 2).astype("int32"), 1)
        score = float(cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], m2)[0])
        if box_thresh > score:
            continue
        expanded = _unclip(points, unclip_ratio)
        if len(expanded) > 1:
            continue
        box2 = np.array(expanded).reshape(-1, 1, 2)
        box2, sside = _get_mini_boxes(box2)
        if sside < min_size + 2:
            continue
        box2 = np.array(box2, dtype="int32")
        box2[:, 0] = np.clip(np.round(box2[:, 0] / w * dest_w), 0, dest_w)
        box2[:, 1] = np.clip(np.round(box2[:, 1] / h * dest_h), 0, dest_h)
        boxes.append(box2)
        scores.append(score)
    return boxes, scores

# Load models
det_sess = ort.InferenceSession(os.path.join(det_dir, "inference.onnx"), providers=["CPUExecutionProvider"])
rec_sess = ort.InferenceSession(os.path.join(rec_dir, "inference.onnx"), providers=["CPUExecutionProvider"])

# Load dict
def _unquote(item):
    s = item.strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1].replace("''", "'")
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s

def load_dict(yml_path):
    lines = open(yml_path, encoding="utf-8").read().splitlines()
    start = next(n for n, l in enumerate(lines) if "character_dict:" in l)
    items, n = [], start + 1
    while n < len(lines):
        m = re.match(r"\s*-(.*)", lines[n])
        if not m:
            break
        items.append(_unquote(m.group(1)))
        n += 1
    return items

import re
labels = ["blank"] + load_dict(os.path.join(rec_dir, "inference.yml")) + [" "]

def recognize_line(img_rgb, sess):
    img_bgr = img_rgb[:, :, ::-1]
    h, w = img_bgr.shape[:2]
    new_w = max(1, min(int(round(w / h * 48)), 320))
    im = cv2.resize(img_bgr, (new_w, 48), interpolation=cv2.INTER_LINEAR)
    arr = (im.astype(np.float32) / 255.0 - 0.5) / 0.5
    if new_w < 320:
        arr = np.concatenate([arr, np.zeros((48, 320 - new_w, 3), np.float32)], axis=1)
    in_name = sess.get_inputs()[0].name
    probs = sess.run(None, {in_name: arr.transpose(2, 0, 1)[None]})[0][0]
    idx = probs.argmax(axis=1)
    chars, prev = [], -1
    for i in idx:
        if i != prev and i != 0:
            chars.append(labels[i])
        prev = i
    merged = [c for i, c in enumerate(chars) if i == 0 or c != chars[i - 1]]
    text = "".join(merged)
    score = float(probs.max(axis=1).mean())
    return text, score

def order_quad(pts):
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl])

def warp_quad(img, quad):
    q = order_quad(quad)
    W = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2])))
    H = max(int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])), 8))
    if W < 2:
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (W, H))

def main():
    path = r"D:\code\zmxy\教师节.png"
    img_bgr = imread_unicode(path)
    img_rgb = img_bgr[:, :, ::-1]
    h, w = img_bgr.shape[:2]

    # Det
    scale = min(960 / max(h, w), 1.0)
    nh = max(32, (int(h * scale) // 32) * 32)
    nw = max(32, (int(w * scale) // 32) * 32)
    im = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = im.astype(np.float32)
    x = (x / 255.0 - _MEAN) / _STD
    prob = det_sess.run(None, {det_sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    boxes, scores = db_postprocess(prob, w, h, box_thresh=0.4)

    print(f"det: {len(boxes)} boxes")

    results = []
    for quad, score in zip(boxes, scores):
        strip = warp_quad(img_rgb, quad)
        if strip is None:
            continue
        text, rec_score = recognize_line(strip, rec_sess)
        x0 = int(quad[:, 0].min())
        y0 = int(quad[:, 1].min())
        x1 = int(quad[:, 0].max())
        y1 = int(quad[:, 1].max())
        results.append({
            "box": [x0, y0, x1, y1],
            "quad": quad.tolist(),
            "det_score": float(score),
            "text": text,
            "rec_score": float(rec_score)
        })
        print(f"box {results[-1]['box']}, text: {text}")
    out = {
        "image_path": path,
        "image_size": [w, h],
        "det_output_size": [nw, nh],
        "results": results
    }
    json.dump(out, open(r"D:\code\zmxy\ppocrv6-min-node\python_ref.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("written python_ref.json")

if __name__ == "__main__":
    main()
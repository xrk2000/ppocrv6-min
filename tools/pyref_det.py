import argparse
import json
import os
import sys
import numpy as np
import cv2
import onnxruntime as ort
import pyclipper

# Path resolution
_LOCAL_ROOT = r"D:\models\PP-OCRv6"

def find_model_dir(env_var, local_name):
    env = os.environ.get(env_var)
    if env and os.path.isfile(os.path.join(env, "inference.onnx")):
        return env
    local = os.path.join(_LOCAL_ROOT, local_name)
    if os.path.isfile(os.path.join(local, "inference.onnx")):
        return local
    try:
        import ppocrv6_min
        cand = ppocrv6_min.BUNDLED_DET_DIR if env_var == "PPOCR_DET_DIR" else ppocrv6_min.BUNDLED_REC_DIR
        if os.path.isfile(os.path.join(cand, "inference.onnx")):
            return cand
    except ImportError:
        pass
    raise FileNotFoundError(f"model not found: {local}")

# Use bundled models by default
det_dir = find_model_dir("PPOCR_DET_DIR", "PP-OCRv6_tiny_det_onnx")

def imread_unicode(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def db_postprocess(pred, dest_w, dest_h, thresh=0.2, box_thresh=0.4, unclip_ratio=1.4, min_size=3):
    mask = (pred > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = pred.shape
    boxes, scores = [], []

    # Dump contour info for comparison
    contour_info = []

    for i, contour in enumerate(contours[:3000]):
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
        points2 = [points[i1], points[i2], points[i3], points[i4]]
        sside = min(bounding_box[1])

        points_list = [[float(p[0]), float(p[1])] for p in points2]

        contour_info.append({
            "id": i,
            "points": points_list,
            "sside": float(sside)
        })

        if sside < min_size:
            continue
        points3 = np.array(points2)
        box = points3.astype(np.float32).copy()
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        m2 = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(m2, box.reshape(1, -1, 2).astype("int32"), 1)
        score = float(cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], m2)[0])

        contour_info[-1]["score"] = float(score)

        if box_thresh > score:
            continue

        # unclip
        box4 = np.asarray(points3, dtype=np.float64)
        x = box4[:, 0]
        y = box4[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        closed = np.vstack([box4, box4[:1]])
        length = float(np.linalg.norm(np.diff(closed, axis=1), axis=1).sum())
        distance = area * unclip_ratio / (length + 1e-9)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box4, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)

        contour_info[-1]["distance"] = float(distance)
        contour_info[-1]["unclip_paths"] = len(expanded)

        if len(expanded) > 1:
            continue

        box5 = np.array(expanded).reshape(-1, 1, 2)
        bounding_box2 = cv2.minAreaRect(box5)
        points6 = sorted(list(cv2.boxPoints(bounding_box2)), key=lambda x: x[0])
        i1, i2, i3, i4 = 0, 1, 2, 3
        if points6[1][1] > points6[0][1]:
            i1, i4 = 0, 1
        else:
            i1, i4 = 1, 0
        if points6[3][1] > points6[2][1]:
            i2, i3 = 2, 3
        else:
            i2, i3 = 3, 2
        box6 = [points6[i1], points6[i2], points6[i3], points6[i4]]
        sside2 = min(bounding_box2[1])

        if sside2 < min_size + 2:
            continue

        box7 = np.array(box6, dtype="int32")
        box7[:, 0] = np.clip(np.round(box7[:, 0] / w * dest_w), 0, dest_w)
        box7[:, 1] = np.clip(np.round(box7[:, 1] / h * dest_h), 0, dest_h)
        boxes.append(box7)
        scores.append(score)

    return boxes, scores, contour_info

def main():
    path = r"D:\code\zmxy\教师节.png"
    img_bgr = imread_unicode(path)
    h, w = img_bgr.shape[:2]

    # Det inference
    det_sess = ort.InferenceSession(os.path.join(det_dir, "inference.onnx"),
                                     providers=["CPUExecutionProvider"])
    in_name = det_sess.get_inputs()[0].name

    limit_side_len = 960
    scale = min(limit_side_len / max(h, w), 1.0)
    nh = max(32, (int(h * scale) // 32) * 32)
    nw = max(32, (int(w * scale) // 32) * 32)

    im = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = im.astype(np.float32)
    x = (x / 255.0 - _MEAN) / _STD
    prob = det_sess.run(None, {in_name: x.transpose(2, 0, 1)[None]})[0][0, 0]

    # Save prob map
    np.save("D:/code/zmxy/ppocrv6-min-node/prob_map.npy", prob)

    # Run DB postprocess
    boxes, scores, contour_info = db_postprocess(prob, w, h)

    # Dump results
    results = {
        "image_size": [int(w), int(h)],
        "det_output_size": [int(nw), int(nh)],
        "prob_shape": [int(s) for s in prob.shape],
        "contours": contour_info,
        "final_boxes": [[[float(p[0]), float(p[1])] for p in box] for box in boxes],
        "final_scores": [float(s) for s in scores]
    }

    with open("D:/code/zmxy/ppocrv6-min-node/python_det_ref.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Contours: {len(contour_info)}")
    print(f"Final boxes: {len(boxes)}")
    print("written python_det_ref.json")

if __name__ == "__main__":
    main()
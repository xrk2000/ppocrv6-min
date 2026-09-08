# ppocrv6-min

**PP-OCRv6 (tiny det + tiny rec) 的最小推理环境** — 只用 4 个 pip 包，
不依赖 PaddlePaddle / PaddleOCR / torch。

[PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) 是 PaddlePaddle 的轻量
OCR 系列（1.5M–34.5M 参数，支持 49+ 语言）。本项目把其中 `tiny` 档的检测
（DB）与识别（CTC）ONNX 模型封装成零框架依赖的 Python 库：

```
numpy + onnxruntime + opencv-python + pyclipper
```

实测（CPU, Windows）：362×215 截图 6 行中文全图识别 ~0.5 s，其中
det 9 ms、rec 1–2 ms/行，识别结果全部正确（置信度 0.96–0.999）。

## 特性

- **4 个依赖**：无需 PaddlePaddle、PaddleOCR、Pillow、shapely、ultralytics
- **后处理与 PaddleOCR 官方逐行对齐**：DB `minAreaRect` 打分 +
  `pyclipper` unclip（`distance = area·ratio/length`），CTC 贪心解码
- **参数可配**：`box_thresh`、`limit_side_len`、onnxruntime providers（可换 GPU）
- **中文字典自动解析**：直接从模型目录 `inference.yml` 的 `character_dict`
  读取（6904 字符，无需单独 dict 文件，无 YAML 库）
- **中文路径支持**：`np.fromfile + cv2.imdecode`，Windows 下中文文件名无问题
- **CLI**：`ppocrv6-min img.png --json out.json`

## 安装

```bash
pip install .            # 从本目录安装
# 或
pip install -e .[test]   # 开发 + 测试
```

依赖（Python 3.9–3.13）：

| 包 | 版本 |
|---|---|
| numpy | >=1.24 |
| onnxruntime | >=1.16（GPU 用 `onnxruntime-gpu`） |
| opencv-python | >=4.8 |
| pyclipper | >=1.3 |

## 模型

从 HuggingFace `PaddlePaddle` 组织下载两个 ONNX 目录（各含
`inference.onnx` + `inference.yml`，rec 的 yml 含字符字典，**必须保留**）：

- [PP-OCRv6_tiny_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx)
- [PP-OCRv6_tiny_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx)

默认查找 `~/models/PP-OCRv6/...`，可用环境变量或参数覆盖：

```bash
export PPOCR_DET_DIR=/path/to/PP-OCRv6_tiny_det_onnx
export PPOCR_REC_DIR=/path/to/PP-OCRv6_tiny_rec_onnx
```

> 目录里的 `inference.json` 是 Paddle PIR 格式权重，onnxruntime 用不到，可忽略。
> 换 medium/small 档位只需把 `det_dir`/`rec_dir` 指到对应目录（medium/small
> 为 18710 类，字典在各自 `inference.yml`，接口不变）。

## 用法

### Python API

```python
from ppocrv6_min import OCR

ocr = OCR("models/PP-OCRv6_tiny_det_onnx",
          "models/PP-OCRv6_tiny_rec_onnx",
          box_thresh=0.4)

items = ocr.predict("截图.png")        # 路径或 BGR ndarray 均可
for it in items:
    print(it.box, f"{it.score:.3f}", it.text)
```

`TextItem` 字段：`text`（识别文本）、`score`（识别置信度）、
`box`（轴对齐框 `(x0,y0,x1,y1)`）、`quad`（四点框，原图坐标）。

单模块使用：

```python
from ppocrv6_min import DetModel, RecModel
from ppocrv6_min.io import imread_unicode
from ppocrv6_min.pipeline import warp_quad

det = DetModel(det_dir)
rec = RecModel(rec_dir)
img = imread_unicode("a.png")          # BGR
for quad, score in det.detect(img):
    strip = warp_quad(img[:, :, ::-1], quad)   # 透视矫正成水平条带 (RGB)
    text, rec_score, _dt, _sz = rec.predict_line(strip)
```

### CLI

```bash
ppocrv6-min screenshot.png                 # det + rec
ppocrv6-min --no-det single_line.png       # 跳过检测，整图当一行识别
ppocrv6-min --box-thresh 0.3 --json out.json a.png b.png
ppocrv6-min --limit-side 736 photo.jpg     # 小图提速
```

## 运行测试

```bash
pytest
```

单元测试（字典解析、unclip、四点排序、warp）无需模型文件；集成测试需要
真实 ONNX 模型，模型不存在时自动 skip。可用环境变量指向模型：

```bash
PPOCR_TEST_DET_DIR=... PPOCR_TEST_REC_DIR=... PPOCR_TEST_IMAGE=... pytest
```

## 项目结构

```
ppocrv6-min/
├── pyproject.toml            # 打包 + 依赖 + CLI 入口
├── src/ppocrv6_min/
│   ├── __init__.py           # 导出 OCR / DetModel / RecModel
│   ├── io.py                 # 中文路径安全的 imread/imwrite
│   ├── det.py                # DB 检测（官方后处理 + pyclipper unclip）
│   ├── rec.py                # CTC 识别（字典解析 + 预处理 + 解码）
│   ├── pipeline.py           # OCR = det → warp → rec
│   └── cli.py                # ppocrv6-min 命令行
├── tests/
│   ├── conftest.py           # 模型路径定位 + skip 逻辑
│   ├── test_units.py         # 无模型依赖的单测
│   └── test_integration.py   # 真实模型端到端（缺模型自动 skip）
├── README.md
└── LICENSE                   # Apache-2.0
```

## 实现要点（踩坑记录）

1. **rec 的 ONNX 输出已是 softmax 概率**（行和 = 1），不能再做第二次
   softmax，否则置信度被压成 ~0.0004。
2. **unclip 必须用官方算法**：先对 `minAreaRect` 在概率图上 `fillPoly`
   打分，过 `box_thresh` 后再用 pyclipper 外扩。用 `cv2.distanceTransform`
   近似是错的（cv2 只能算 0 像素到最近 1 像素的距离，方向与 DB 需要的
   文字像素膨胀相反）。
3. **`cv2.imread` 在 Windows 读不了中文路径**，用 `np.fromfile + cv2.imdecode`。
4. det 框分数异常低（<0.1）时，先检查是否在**缩放后**的 prob 图上用了
   **原图坐标**采样（坐标空间不一致）。
5. rec 输入是"单行文字条带"：整图直接喂只会得到低置信乱码，必须先 det 切行。
6. 四点框排序用 sum/diff 角点法得到 tl/tr/br/bl；条带高度 <8 px 时放大到
   8 px，避免识别崩坏。
7. yml 里的 YAML 引号字符（`''''` = 单引号、`'"'` = 双引号）只需剥掉外层
   一对引号，无需完整 YAML 解析器。

## 许可

Apache-2.0。模型权重许可见对应 HuggingFace 仓库（apache-2.0）。

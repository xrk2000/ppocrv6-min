# ppocrv6-min

**PP-OCRv6 (tiny det + tiny rec) 的最小推理环境** — 只用 3 个 npm 包，
不依赖 PaddlePaddle / PaddleOCR / OpenCV / Python。

[PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) 是 PaddlePaddle 的轻量
OCR 系列（1.5M–34.5M 参数，支持 49+ 语言）。本项目把其中 `tiny` 档的检测
（DB）与识别（CTC）ONNX 模型封装成零框架依赖的 Node.js 库：

```
onnxruntime-node + pngjs + jpeg-js
```

实测（CPU, Windows）：362×215 截图 6 行中文全图识别 ~100 ms，其中
det ~60 ms、rec 3-5 ms/行，识别结果全部正确（置信度 0.96–0.998）。

## 特性

- **3 个依赖**：无需 PaddlePaddle、PaddleOCR、OpenCV、Python
- **后处理与 PaddleOCR 官方逐行对齐**：DB `minAreaRect` 打分 +
  miter offset unclip（`distance = area·ratio/length`），CTC 贪心解码
- **参数可配**：`box_thresh`、`limit_side_len`、onnxruntime providers（可换 GPU）
- **中文字典自动解析**：直接从模型目录 `inference.yml` 的 `character_dict`
  读取（6904 字符，无需单独 dict 文件，无 YAML 库）
- **CLI**：`ppocrv6-min img.png --json out.json`

## 安装

```bash
npm install ppocrv6-min   # 从 npm 安装
npm install .             # 或从本目录安装（开发用）
```

依赖（Node.js >= 18）：

| 包 | 版本 |
|---|---|
| onnxruntime-node | >=1.23.0（GPU 用 onnxruntime-node-gpu） |
| pngjs | >=7.0.0 |
| jpeg-js | >=0.4.4 |

## 模型

**`tiny` 档模型已随包内置**（det ~1.7 MB + rec ~4.4 MB，位于
`models/`），`npm install` 之后**开箱即用，无需下载任何权重**。

来源（Apache-2.0，如需换档位从这里下载）：

- [PP-OCRv6_tiny_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx)
- [PP-OCRv6_tiny_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx)

默认使用内置模型；要换 small/medium 档位（18710 类，字典在各自
`inference.yml`，接口不变）时，用参数或环境变量指向：

```bash
export PPOCR_DET_DIR=/path/to/PP-OCRv6_small_det_onnx
export PPOCR_REC_DIR=/path/to/PP-OCRv6_small_rec_onnx
```

## 用法

### JavaScript API

```javascript
import { OCR } from "ppocrv6-min";

const ocr = await new OCR().init();      // 加载 ONNX 会话（一次）
const items = await ocr.predict("截图.png");
for (const it of items) {
  console.log(it.box, it.score.toFixed(3), it.text);
}
```

`TextItem` 字段：`text`（识别文本）、`score`（识别置信度）、
`box`（轴对齐框 `[x0,y0,x1,y1]`）、`quad`（四点框，原图坐标）。

单模块使用：

```javascript
import { DetModel, RecModel, imread, toRgb, warpQuad } from "ppocrv6-min";

const det = new DetModel(detDir);
await det.init();

const rec = new RecModel(recDir);
await rec.init();

const img = imread("a.png"); // BGR
const quadsScores = await det.detect(img);

const imgRgb = toRgb(img);
for (const { quad } of quadsScores) {
  const strip = warpQuad(imgRgb, quad);
  const { text, score } = await rec.predictLine(strip);
  console.log(text, score);
}
```

### CLI

```bash
ppocrv6-min screenshot.png                 # det + rec
ppocrv6-min --no-det single_line.png       # 跳过检测，整图当一行识别
ppocrv6-min --box-thresh 0.3 --json out.json a.png b.png
ppocrv6-min --limit-side 736 photo.jpg     # 小图提速
```

> 本包是纯 ESM（`"type": "module"`）：请用 `import`，CommonJS 的 `require`
> 不支持。

## 测试

```bash
npm test
```

零测试框架依赖（`node tests/run_tests.mjs`）。单元测试（字典解析、unclip、
四点排序、warp、cv 原语、PNG 读写）不需要模型文件；集成测试使用内置 tiny
模型 + `tests/data/sample.png`（一行中文"木灵葫芦"），模型缺失时自动跳过。
可用环境变量指向其他模型/图片：

```bash
PPOCR_TEST_DET_DIR=... PPOCR_TEST_REC_DIR=... PPOCR_TEST_IMAGE=... npm test
```

## 项目结构

```
ppocrv6-min-node/
├── package.json            # 打包 + 依赖 + CLI 入口（files: src + models）
├── src/
│   ├── index.js            # 主导出（OCR, DetModel, RecModel, 常量）
│   ├── io.js              # 图像解码（PNG/JPEG -> BGR/RGB）
│   ├── paths.js           # 内置模型目录解析（env 覆盖）
│   ├── det.js             # DB 检测（官方后处理 + miter offset unclip）
│   ├── rec.js             # CTC 识别（字典解析 + 预处理 + 解码）
│   ├── pipeline.js        # OCR = det → warp → rec
│   ├── cli.js             # ppocrv6-min 命令行
│   └── cv.js              # OpenCV 风格操作（resize, contours, 几何, warp）
├── models/                # ★ 内置 tiny 模型（随 npm 分发）
│   ├── PP-OCRv6_tiny_det_onnx/   inference.onnx + inference.yml
│   └── PP-OCRv6_tiny_rec_onnx/   inference.onnx + inference.yml
├── tests/
│   ├── run_tests.mjs      # 零依赖测试运行器（npm test）
│   ├── units.mjs          # 无模型依赖的单测
│   ├── integration.mjs    # 内置模型端到端（可 skip）
│   └── data/sample.png    # 集成测试用单行中文图
├── tools/                 # 本地调试/对拍脚本（不进 npm 包，路径需本地修改）
├── README.md
└── LICENSE                # Apache-2.0
```

## 实现要点（踩坑记录）

1. **rec 的 ONNX 输出已是 softmax 概率**（行和 = 1），不能做第二次 softmax
2. **unclip 使用 miter offset**：对于 minAreaRect 的矩形框，miter offset 等价于 pyclipper 的 JT_ROUND offset（矩形外扩后 minAreaRect 相同）
3. **findContours 使用连通分量 + 边界追踪**：避免了 Suzuki-Abe 复杂实现的边界情况，结果更稳定
4. **orderQuad 的 diff 计算**：应该是 `y - x`（即 `np.diff(pts, axis=1)[:, 0]`），错误地用 `x - y` 会导致四边形排序错误（造成奇异矩阵）
5. **resize 双线性插值**：使用 OpenCV 的中心对齐公式 `dst = (src + 0.5) * scale - 0.5`
6. **warpPerspective 边界处理**：超出边界的像素采样为 0（BORDER_CONSTANT）
7. **yml 引号字符解析**：`''''` → `'`, `'"'` → `"`（只需剥掉外层引号）
8. **孔洞边界的扫描方向**：孔洞边界需逆时针扫描（与外边界相反），否则会产生重复轮廓

## 许可

Apache-2.0。模型权重许可见对应 HuggingFace 仓库（apache-2.0）。
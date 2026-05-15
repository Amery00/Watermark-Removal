# 水印去除与图像复刻工具链 (Watermark Removal & Image Replication)
## 1. 项目概述

本项目是一个**全栈水印去除与图像复刻系统**，核心能力包括：

- **智能复刻**：基于参考图生成风格一致的清洁版本
- **水印自动检测**：OCR + 颜色分析 + Logo 匹配，自动定位水印区域
- **水印去除**：AI 复刻图融合 + OpenCV Inpainting，支持自动/手动双模式
- **可视化 GUI**：零依赖桌面应用，实时预览与参数调节
- **RESTful API**：FastAPI 构建，支持 HTTP 调用、批量处理、多种返回格式

---

## 2. 技术选型

### 2.1 Python 依赖栈

| 层级 | 库 | 版本 | 用途 |
|:---|:---|:---|:---|
| 图像解码 | Pillow + pillow-heif | >=10.0 | 多格式读取，HEIC 自动降级 |
| 矩阵运算 | NumPy | >=1.24 | 像素级数组操作 |
| 图像处理 | OpenCV-Python | >=4.8 | Inpainting、颜色空间转换、形态学 |
| 科学滤波 | SciPy | >=1.10 | 高斯模糊、平滑过渡 |
| OCR (可选) | easyocr / pytesseract | - | 水印文字识别 |
| Web 框架 | FastAPI | >=0.100 | REST API 服务 |
| 服务器 | uvicorn | >=0.23 | ASGI 运行环境 |
| 可视化 | Matplotlib | >=3.7 | CLI 调试预览（可选） |

### 2.2 GUI 技术

- **tkinter** — Python 标准库，零额外依赖
- **ttk + 自定义主题** — 现代化样式
- **PIL.ImageTk** — Canvas 图像渲染

### 2.3 选型理由

- **纯 CV 优先检测**：默认使用 OpenCV 亮度分析（毫秒级），OCR 作为可选增强
- **FastAPI 而非 Flask**：原生异步支持、自动 OpenAPI/Swagger 文档、类型安全
- **tkinter 而非 PyQt**：零安装负担，跨平台一致

---

## 3. 系统架构

```
┌─────────────────────────────────────────────┐
│                 用户接口层                    │
│  ┌─────────┐  ┌─────────┐  ┌───────────────┐ │
│  │ 命令行CLI│  │ 桌面GUI │  │ REST API      │ │
│  │         │  │         │  │ (FastAPI)     │ │
│  └─────────┘  └─────────┘  └───────────────┘ │
│       │            │              │          │
└───────┼────────────┼──────────────┼──────────┘
        │            │              │
        └────────────┴──────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│                 核心处理层                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │ 自动检测模块 │ │ 水印去除引擎 │ │ 图像解码 │ │
│  │watermark_   │ │watermark_   │ │ImageDec │ │
│  │detector.py  │ │remover.py   │ │oder     │ │
│  └─────────────┘ └─────────────┘ └─────────┘ │
│       ▲                ▲                      │
│       └────────────────┘                      │
│           OpenCV / NumPy / SciPy              │
└─────────────────────────────────────────────┘
```

---

## 4. 模块设计

### 4.1 `watermark_detector.py` — 自动检测模块

**策略优先级**：

1. **颜色异常分析**（默认启用）：检测底部白色/亮色文字，毫秒级
2. **Logo 几何检测**（默认启用）：检测左下角圆形/方形图标
3. **OCR 文字检测**（可选）：easyocr / pytesseract，较慢但更精准

**核心类**：

- `WatermarkDetector(use_ocr=False)` — 主检测器
  - `detect(image)` → `(mask, meta)`
  - `detect_params(image)` → 建议参数字典
- `auto_detect_watermark(image_path)` — 便捷函数

### 4.2 `watermark_remover.py` — 去除引擎

**核心类**：

- `ImageDecoder` — 多格式解码（HEIC 降级）
- `WatermarkRemover` — 去除引擎
  - `remove_bottom_strip()` — 复刻图替换或 Inpainting 降级
  - `inpaint_region()` — OpenCV 精修
  - `texture_synthesis_fill()` — 纹理合成平滑

**主函数**：

- `process_image(..., auto_detect=False)` — 完整流水线

### 4.3 `watermark_gui.py` — 可视化桌面应用

**功能**：

- 文件选择（原图 / 复刻图）
- **自动检测开关**：勾选后自动识别水印位置，参数滑块变为只读展示
- **OCR 增强开关**：检测时启用文字识别
- 实时预览（原图 / 结果 / 对比）
- 底部红色虚线标注水印区域
- **API 服务启动按钮**：在 GUI 内一键启动 REST API

### 4.4 `api_server.py` — REST API 服务

| 方法 | 路径 | 功能 |
|:---|:---|:---|
| GET | `/api/v1/health` | 健康检查 |
| POST | `/api/v1/detect` | 水印检测，返回建议参数 |
| POST | `/api/v1/remove` | 手动参数去除 |
| POST | `/api/v1/remove-auto` | 全自动去除（检测 + 处理） |
| POST | `/api/v1/batch-detect` | 批量检测（实验性） |

**返回类型**（`return_type` 参数控制）：

- `file` — 直接下载处理后的图像
- `base64` — JSON 内嵌 Base64 图像
- `info` — 仅返回处理信息（尺寸、参数、耗时等）

---

## 5. 三种使用模式

### 5.1 命令行模式 (CLI)

```bash
# 全自动（自动检测水印位置）
python watermark_remover.py -i input.png --auto-detect -o clean.png

# 手动参数（指定复刻图）
python watermark_remover.py -i input.png -r replica.png -o clean.png
```

### 5.2 可视化 GUI 模式

```bash
python watermark_gui.py
```

**GUI 操作步骤**：

1. 勾选「自动检测水印」
2. 选择原图（复刻图可选）
3. 点击「检测水印位置」预览自动参数
4. 点击「开始处理」
5. 预览对比，满意后保存
6. （可选）点击「启动 API 服务」开启 REST 接口

### 5.3 API 服务模式

```bash
# 启动服务
python api_server.py --host 0.0.0.0 --port 8000

# 访问 Swagger 文档
open http://localhost:8000/api/v1/docs
```

**cURL 调用示例**：

```bash
# 1. 检测水印位置
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "image=@photo.jpg"

# 2. 全自动去除（强烈推荐上传复刻图）
curl -X POST "http://localhost:8000/api/v1/remove-auto" \
  -F "image=@photo.jpg" \
  -F "replica=@ai_replica.png" \
  --output clean.png

# 3. 获取 base64 结果（用于前端直显）
curl -X POST "http://localhost:8000/api/v1/remove-auto" \
  -F "image=@photo.jpg" \
  -F "return_type=base64"

# 4. 仅获取处理信息
curl -X POST "http://localhost:8000/api/v1/remove-auto" \
  -F "image=@photo.jpg" \
  -F "return_type=info"
```

---

## 6. 文件清单

| 文件 | 说明 |
|:---|:---|
| `watermark_remover.py` | 核心处理引擎 + CLI |
| `watermark_detector.py` | 水印自动检测模块 |
| `watermark_gui.py` | 可视化桌面应用 |
| `api_server.py` | FastAPI REST 服务 |
| `requirements.txt` | 依赖列表 |

---

## 7. 安装与运行

```bash
# 安装基础依赖
pip install -r requirements.txt

# 可选：OCR 增强（较慢但更精准）
pip install easyocr

# Linux 需安装系统 tk 支持
sudo apt-get install python3-tk
```

---

## 8. 注意事项

1. **AI 复刻图依赖**：核心效果依赖复刻图质量。无复刻图  时降级为 OpenCV Inpainting，仅适合小水印。
2. **OCR 速度**：easyocr 首次运行需下载模型（~100MB），加载时间约 5–10 秒。
3. **HEIC 兼容**：Linux 需 `sudo apt-get install libheif-examples` 或 FFmpeg。
4. **API 多进程**：生产环境建议 `--workers 4`，开发模式用 `--reload`。

---

## License

本项目在 **MIT License**（麻省理工学院许可证）下发布。仓库根目录的 [`LICENSE`](LICENSE) 与下文一致；亦可参阅 [Open Source Initiative 上的 MIT 说明](https://opensource.org/licenses/MIT)。

```
MIT License

Copyright (c) 2026 Watermark Removal & Image Replication Project contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

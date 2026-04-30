#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watermark Remover API Server
============================
基于 FastAPI 的 RESTful API 服务，提供水印检测与去除能力。

端点:
    POST /api/v1/detect    — 上传图像，返回水印检测结果与建议参数
    POST /api/v1/remove    — 上传原图+可选复刻图，返回处理后的无水印图像
    POST /api/v1/remove-auto — 自动检测水印位置并去除（无需手动参数）
    GET  /api/v1/health    — 健康检查
    GET  /api/v1/docs      — API 文档 (Swagger UI)

启动:
    python api_server.py [--host 0.0.0.0] [--port 8000] [--reload]

Usage (curl 示例):
    # 检测水印
    curl -X POST "http://localhost:8000/api/v1/detect" \\
         -F "image=@/path/to/photo.jpg"

    # 去除水印（自动模式）
    curl -X POST "http://localhost:8000/api/v1/remove-auto" \\
         -F "image=@/path/to/photo.jpg" \\
         --output cleaned.png

    # 去除水印（带复刻图）
    curl -X POST "http://localhost:8000/api/v1/remove" \\
         -F "original=@/path/to/photo.jpg" \\
         -F "replica=@/path/to/replica.png" \\
         -F "strip_height=90" \\
         -F "blend_zone=20" \\
         --output cleaned.png
"""

import io
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 导入核心模块
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from watermark_remover import process_image, ImageDecoder
    from watermark_detector import WatermarkDetector, auto_detect_watermark, WatermarkMeta
except ImportError as e:
    print(f"[错误] 无法导入核心模块: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Watermark Remover API",
    description="智能水印检测与去除 RESTful 服务",
    version="1.1.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class DetectResponse(BaseModel):
    success: bool
    detected: bool
    confidence: float
    suggested_params: Dict[str, int]
    texts: list
    logo: Optional[str]
    method: str
    message: str


class RemoveResponse(BaseModel):
    success: bool
    download_url: Optional[str]
    message: str
    params_used: Optional[Dict[str, Any]]
    processing_time_ms: Optional[int]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    features: list


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _save_upload(upload: UploadFile) -> str:
    """将上传文件保存到临时路径，返回路径"""
    suffix = Path(upload.filename).suffix or ".tmp"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """将 PIL 图像转为字节流"""
    buf = io.BytesIO()
    if fmt.upper() in ("JPEG", "JPG"):
        img = img.convert("RGB")
    img.save(buf, format=fmt.upper())
    buf.seek(0)
    return buf.getvalue()


def _cleanup(*paths: str):
    """清理临时文件"""
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def _normalize_response_options(output_format: str, return_type: str) -> tuple[str, str]:
    fmt = (output_format or "png").lower().strip()
    rtype = (return_type or "file").lower().strip()
    if fmt not in ("png", "jpg", "jpeg"):
        raise HTTPException(status_code=400, detail="output_format 仅支持 png/jpg/jpeg")
    if rtype not in ("file", "base64", "info"):
        raise HTTPException(status_code=400, detail="return_type 仅支持 file/base64/info")
    return ("jpg" if fmt == "jpeg" else fmt), rtype


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
def root():
    """API 根路径，返回基本信息"""
    return {
        "service": "Watermark Remover API",
        "version": "1.1.0",
        "docs": "/api/v1/docs",
        "endpoints": {
            "detect": "POST /api/v1/detect",
            "remove": "POST /api/v1/remove",
            "remove_auto": "POST /api/v1/remove-auto",
            "health": "GET /api/v1/health",
        },
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
def health():
    """健康检查端点"""
    return HealthResponse(
        status="ok",
        version="1.1.0",
        timestamp=datetime.now().isoformat(),
        features=["watermark_detection", "watermark_removal", "auto_removal", "replica_blend", "ai_replica_generation"],
    )


@app.post(
    "/api/v1/detect",
    response_model=DetectResponse,
    tags=["Detection"],
    summary="水印自动检测",
)
def api_detect(image: UploadFile = File(..., description="待检测的图像文件")):
    """
    上传图像，自动检测其中的水印位置、文字内容和置信度。
    返回建议的处理参数（strip_height, blend_zone 等）。
    """
    tmp_path = None
    try:
        # 保存上传文件
        tmp_path = _save_upload(image)

        # 执行检测
        result = auto_detect_watermark(tmp_path)
        return DetectResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")
    finally:
        _cleanup(tmp_path)


@app.post(
    "/api/v1/remove",
    tags=["Removal"],
    summary="水印去除（手动参数）",
    response_description="处理后的无水印图像（PNG格式）",
)
def api_remove(
    original: UploadFile = File(..., description="原始有水印图像"),
    replica: Optional[UploadFile] = File(None, description="AI复刻清洁图像（可选）"),
    strip_height: int = Form(90, description="底部替换高度（像素）"),
    blend_zone: int = Form(20, description="顶部过渡融合区（像素）"),
    corner_blur: float = Form(2.0, description="左下角模糊度"),
    output_format: str = Form("png", description="输出格式: png 或 jpg"),
    return_type: str = Form("file", description="返回类型: file=直接下载, base64=JSON含base64, info=仅返回信息"),
    ai_replicate: bool = Form(False, description="未提供replica时是否调用AI生成复刻图"),
    ai_api_base_url: str = Form("", description="AI接口基础地址，如 https://api.openai.com/v1"),
    ai_api_key: str = Form("", description="AI接口密钥"),
    ai_model: str = Form("auto", description="AI图像模型名，默认auto自动识别"),
    ai_prompt: str = Form("保留主体构图和风格，完整去除所有底部水印、角标与文字标识，输出自然无痕迹图像。", description="AI复刻图提示词"),
):
    """
    上传原图（和可选复刻图），根据指定参数去除水印。
    直接返回处理后的图像文件。
    """
    orig_tmp = None
    rep_tmp = None
    out_tmp = None

    try:
        output_format, return_type = _normalize_response_options(output_format, return_type)
        # 保存上传
        orig_tmp = _save_upload(original)
        rep_tmp = _save_upload(replica) if replica else None

        # 处理
        fd, out_tmp = tempfile.mkstemp(suffix=f".{output_format}")
        os.close(fd)

        result_path = process_image(
            original_path=orig_tmp,
            replica_path=rep_tmp,
            output_path=out_tmp,
            visualize=False,
            strip_height=strip_height,
            blend_zone=blend_zone,
            corner_tex_blur=corner_blur,
            ai_replicate=ai_replicate,
            ai_api_base_url=ai_api_base_url,
            ai_api_key=ai_api_key,
            ai_model=ai_model,
            ai_prompt=ai_prompt,
        )

        if return_type == "info":
            img = Image.open(result_path)
            return JSONResponse({
                "success": True,
                "message": "处理完成",
                "width": img.width,
                "height": img.height,
                "format": output_format,
                "params_used": {
                    "strip_height": strip_height,
                    "blend_zone": blend_zone,
                    "corner_blur": corner_blur,
                },
            })

        if return_type == "base64":
            import base64
            with open(result_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return JSONResponse({
                "success": True,
                "image_base64": data,
                "format": output_format,
                "message": "处理完成",
            })

        # file: 直接下载
        media_type = "image/jpeg" if output_format.lower() in ("jpg", "jpeg") else "image/png"
        return FileResponse(
            result_path,
            media_type=media_type,
            filename=f"cleaned.{output_format}",
            background=BackgroundTask(_cleanup, result_path),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    finally:
        _cleanup(orig_tmp, rep_tmp)
        # 注意: out_tmp 在 FileResponse 返回后由调用方使用，不能立即删除
        # 如需清理，需用后台任务（此处简化，依赖系统临时文件清理）


@app.post(
    "/api/v1/remove-auto",
    tags=["Removal"],
    summary="水印去除（全自动模式）",
    response_description="处理后的无水印图像（PNG格式）",
)
def api_remove_auto(
    image: UploadFile = File(..., description="原始有水印图像"),
    replica: Optional[UploadFile] = File(None, description="AI复刻清洁图像（可选，强烈建议提供）"),
    output_format: str = Form("png", description="输出格式: png 或 jpg"),
    return_type: str = Form("file", description="返回类型: file=直接下载, base64=JSON含base64, info=仅返回信息"),
    ai_replicate: bool = Form(False, description="未提供replica时是否调用AI生成复刻图"),
    ai_api_base_url: str = Form("", description="AI接口基础地址，如 https://api.openai.com/v1"),
    ai_api_key: str = Form("", description="AI接口密钥"),
    ai_model: str = Form("auto", description="AI图像模型名，默认auto自动识别"),
    ai_prompt: str = Form("保留主体构图和风格，完整去除所有底部水印、角标与文字标识，输出自然无痕迹图像。", description="AI复刻图提示词"),
):
    """
    **全自动水印去除** — 无需任何手动参数！

    系统会自动：
    1. 检测水印位置（OCR + 颜色分析 + Logo 匹配）
    2. 计算最佳处理参数
    3. 执行去除（若有复刻图则融合，无则纯 Inpainting）
    4. 返回清洁图像

    **强烈建议同时上传 AI 复刻图以获得最佳效果。**
    """
    import time
    start = time.time()

    orig_tmp = None
    rep_tmp = None
    out_tmp = None

    try:
        output_format, return_type = _normalize_response_options(output_format, return_type)
        # 保存上传
        orig_tmp = _save_upload(image)
        rep_tmp = _save_upload(replica) if replica else None

        # 步骤 1: 自动检测
        detector = WatermarkDetector()
        _, meta = detector.detect(orig_tmp)

        # 步骤 2: 执行处理（使用检测到的参数）
        fd, out_tmp = tempfile.mkstemp(suffix=f".{output_format}")
        os.close(fd)

        result_path = process_image(
            original_path=orig_tmp,
            replica_path=rep_tmp,
            output_path=out_tmp,
            visualize=False,
            strip_height=meta.suggested_strip_height,
            blend_zone=meta.suggested_blend_zone,
            corner_tex_blur=2.0,
            ai_replicate=ai_replicate,
            ai_api_base_url=ai_api_base_url,
            ai_api_key=ai_api_key,
            ai_model=ai_model,
            ai_prompt=ai_prompt,
        )

        elapsed = int((time.time() - start) * 1000)
        meta_detected = bool(meta.detected)
        meta_confidence = float(meta.confidence)
        meta_msg = f"检测到水印 (置信度 {meta_confidence:.2f})" if meta_detected else "未检测到明显水印"

        if return_type == "info":
            img = Image.open(result_path)
            return JSONResponse({
                "success": True,
                "message": meta_msg,
                "width": img.width,
                "height": img.height,
                "format": output_format,
                "detected": meta_detected,
                "confidence": meta_confidence,
                "params_used": {
                    "strip_height": int(meta.suggested_strip_height),
                    "blend_zone": int(meta.suggested_blend_zone),
                    "method": str(meta.method),
                },
                "processing_time_ms": elapsed,
            })

        if return_type == "base64":
            import base64
            with open(result_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return JSONResponse({
                "success": True,
                "image_base64": data,
                "format": output_format,
                "message": meta_msg,
                "detected": meta_detected,
                "confidence": meta_confidence,
                "processing_time_ms": elapsed,
            })

        # file 模式：直接下载
        media_type = "image/jpeg" if output_format.lower() in ("jpg", "jpeg") else "image/png"
        return FileResponse(
            result_path,
            media_type=media_type,
            filename=f"cleaned_auto.{output_format}",
            background=BackgroundTask(_cleanup, result_path),
            headers={
                "X-Watermark-Detected": str(meta_detected).lower(),
                "X-Detection-Confidence": f"{meta_confidence:.2f}",
                "X-Processing-Time-Ms": str(elapsed),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    finally:
        _cleanup(orig_tmp, rep_tmp)


# ---------------------------------------------------------------------------
# 批量处理端点（实验性）
# ---------------------------------------------------------------------------

@app.post("/api/v1/batch-detect", tags=["Batch"])
def api_batch_detect(
    images: list[UploadFile] = File(..., description="多张待检测图像"),
):
    """批量水印检测"""
    results = []
    for upload in images:
        tmp = None
        try:
            tmp = _save_upload(upload)
            result = auto_detect_watermark(tmp)
            result["filename"] = upload.filename
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "filename": upload.filename,
                "message": str(e),
            })
        finally:
            _cleanup(tmp)
    return {"success": True, "results": results}


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="启动水印去除 API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="端口 (默认 8000)")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数 (默认 1)")
    args = parser.parse_args()

    print(f"🚀 启动 Watermark Remover API")
    print(f"   地址: http://{args.host}:{args.port}")
    print(f"   文档: http://{args.host}:{args.port}/api/v1/docs")
    print(f"   模式: {'开发(热重载)' if args.reload else '生产'}")
    print(f"   进程: {args.workers}")

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )
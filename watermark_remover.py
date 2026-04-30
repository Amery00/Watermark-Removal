#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watermark Remover & Image Restoration Tool
==========================================
基于AI复刻图与OpenCV Inpainting的智能水印去除工具。

支持HEIC/HEIF/PNG/JPG输入，输出高质量无水印图像。

Usage:
    # 推荐模式：原图 + AI复刻图（效果最佳）
    python watermark_remover.py --original input.heic --replica ai_copy.png --output clean.png

    # 降级模式：仅原图（依赖Inpainting，适合小水印）
    python watermark_remover.py --original input.png --output clean.png

    # 查看调试预览
    python watermark_remover.py --original input.heic --replica ai_copy.png --output clean.png --visualize
"""

import argparse
import base64
import json
import subprocess
import sys
import shutil
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# 0. 依赖检查与友好提示
# ---------------------------------------------------------------------------

def _check_deps():
    """运行时依赖检查，缺失时打印提示并返回 False。"""
    missing = []
    try:
        import PIL
    except ImportError:
        missing.append("Pillow")
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    if missing:
        print("[错误] 缺少以下依赖，请先安装：")
        print(f"    pip install {' '.join(missing)}")
        if "Pillow" in missing:
            print("    # 如需HEIC支持，额外安装：")
            print("    pip install pillow-heif")
        return False
    return True


if __name__ == "__main__" and not _check_deps():
    sys.exit(1)

import cv2

# 尝试注册HEIF解码器，失败不影响PNG/JPG功能
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIF_SUPPORTED = True
except ImportError:
    _HEIF_SUPPORTED = False
    print("[警告] 未安装 pillow-heif，HEIC/HEIF格式将尝试FFmpeg降级解码。")


# ---------------------------------------------------------------------------
# 1. 图像解码模块
# ---------------------------------------------------------------------------

class ImageDecoder:
    """多格式图像解码器，支持HEIC/HEIF降级处理。"""

    @staticmethod
    def load(image_path: str) -> Image.Image:
        """
        读取图像并统一转为RGB模式。
        支持HEIC自动解码（pillow-heif优先，FFmpeg降级）。
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")

        suffix = path.suffix.lower()

        # 尝试标准Pillow读取（已注册HEIF opener时可直接读HEIC）
        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception:
            pass

        # HEIC降级：用FFmpeg转PNG
        if suffix in (".heic", ".heif"):
            print(f"[信息] Pillow无法直接解码，尝试FFmpeg降级: {image_path}")
            temp_png = _temp_path(image_path, ".png")
            if ImageDecoder._heic_via_ffmpeg(image_path, temp_png):
                img = Image.open(temp_png).convert("RGB")
                os.remove(temp_png)
                return img

        raise RuntimeError(f"无法解码图像: {image_path}")

    @staticmethod
    def _heic_via_ffmpeg(src: str, dst: str) -> bool:
        """调用FFmpeg将HEIC转为PNG，返回是否成功。"""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("[警告] 未找到ffmpeg，无法处理HEIC降级。")
            return False
        cmd = [
            ffmpeg, "-y", "-i", src,
            "-frames:v", "1",
            dst
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0 and os.path.exists(dst)
        except Exception as e:
            print(f"[错误] FFmpeg转码失败: {e}")
            return False


# ---------------------------------------------------------------------------
# 2. 水印去除引擎
# ---------------------------------------------------------------------------

class WatermarkRemover:
    """
    水印去除核心引擎。

    策略：
    1. 若提供AI复刻图，将原图底部水印带替换为复刻图对应区域（粗去除）。
    2. 对残留小水印（如AI生成角标）进行OpenCV Inpainting（精修复）。
    3. 高斯滤波平滑拼接边界，消除接缝。
    """

    def __init__(self, original_np: np.ndarray, replica_np: Optional[np.ndarray] = None):
        """
        Args:
            original_np: 原始有水印图 (H, W, 3) uint8 RGB
            replica_np:  AI复刻清洁图 (H, W, 3) uint8 RGB，可为None
        """
        self.orig = original_np.copy()
        self.replica = replica_np
        self.h, self.w = original_np.shape[:2]

    # ------------------------------------------------------------------
    # 2.1 粗去除：底部区域替换
    # ------------------------------------------------------------------
    def remove_bottom_strip(
            self,
            strip_height: int = 90,
            blend_zone: int = 20,
            use_inpaint_fallback: bool = True,
    ) -> np.ndarray:
        """
        去除底部水印带。

        有复刻图时：用复刻图底部替换原图底部。
        无复刻图时：生成底部 mask，用 OpenCV Inpainting 修复（降级方案）。

        Args:
            strip_height: 从底部向上覆盖的像素高度。
            blend_zone:   替换区域顶部渐变过渡带。
            use_inpaint_fallback: 无复刻图时是否使用 Inpainting 降级。

        Returns:
            处理后的图像数组。
        """
        if self.replica is not None:
            # 有复刻图：直接替换
            rep = self._match_size(self.replica, (self.w, self.h))
            result = self.orig.copy()
            y_start = self.h - strip_height
            result[y_start:, :] = rep[y_start:, :]

            # 渐变过渡
            if blend_zone > 0:
                for dy in range(blend_zone):
                    y = y_start + dy
                    if y >= self.h:
                        break
                    alpha = dy / blend_zone
                    result[y, :] = ((1 - alpha) * self.orig[y, :] + alpha * rep[y, :]).astype(np.uint8)
            return result

        # 无复刻图：Inpainting 降级
        if use_inpaint_fallback:
            result = self.orig.copy()
            h, w = result.shape[:2]
            y_start = max(0, h - strip_height)
            mask = np.zeros((h, w), dtype=np.uint8)
            # 对整个底部条带做 mask
            mask[y_start:, :] = 255
            return WatermarkRemover.inpaint_region(result, mask, radius=5, method=cv2.INPAINT_NS)

        print("[信息] 未提供复刻图，跳过底部替换。")
        return self.orig.copy()

    # ------------------------------------------------------------------
    # 2.2 精修复：OpenCV Inpainting 去除残留小水印
    # ------------------------------------------------------------------
    @staticmethod
    def inpaint_region(
            image: np.ndarray,
            mask: np.ndarray,
            radius: int = 3,
            method: int = cv2.INPAINT_TELEA
    ) -> np.ndarray:
        """
        对mask标记的区域进行Inpainting修复。

        Args:
            image: (H, W, 3) uint8 RGB
            mask:  (H, W) uint8，255表示需修复区域
            radius: 修复邻域半径
            method: cv2.INPAINT_TELEA 或 cv2.INPAINT_NS

        Returns:
            修复后的RGB图像
        """
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        repaired = cv2.inpaint(bgr, mask, radius, method)
        return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # 2.3 纹理合成：基于邻近像素填充（备用方案）
    # ------------------------------------------------------------------
    @staticmethod
    def texture_synthesis_fill(
            image: np.ndarray,
            y_top: int,
            y_bottom: int,
            x_left: int,
            x_right: int,
            source_offset_y: int = -60,
            source_offset_x: int = 0,
            blur_sigma: float = 2.0
    ) -> np.ndarray:
        """
        将指定矩形区域用上方/侧方邻近纹理填充，并做高斯平滑。
        用于修复Inpainting后仍存在的局部瑕疵。

        Args:
            image: 原始图像 (H, W, 3) uint8 RGB
            y_top, y_bottom, x_left, x_right: 目标填充区域（闭区间，numpy切片风格）
            source_offset_y: 采样源相对目标的Y偏移（负值=向上取）
            source_offset_x: 采样源相对目标的X偏移
            blur_sigma: 填充后高斯模糊sigma

        Returns:
            填充后的图像
        """
        result = image.copy()
        h, w = result.shape[:2]

        for y in range(y_top, min(y_bottom, h)):
            for x in range(x_left, min(x_right, w)):
                src_y = y + source_offset_y
                src_x = x + source_offset_x
                src_y = max(0, min(src_y, h - 1))
                src_x = max(0, min(src_x, w - 1))
                result[y, x] = image[src_y, src_x]

        # 仅对填充区域做高斯平滑
        patch = result[y_top:y_bottom, x_left:x_right].astype(float)
        patch = gaussian_filter(patch, sigma=(blur_sigma, blur_sigma, 0))
        result[y_top:y_bottom, x_left:x_right] = patch.astype(np.uint8)
        return result

    # ------------------------------------------------------------------
    # 2.4 辅助：尺寸对齐
    # ------------------------------------------------------------------
    @staticmethod
    def _match_size(img_np: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        将numpy图像resize到目标尺寸 (width, height)，LANCZOS插值。
        若尺寸一致则直接返回。
        """
        tw, th = target_size
        h, w = img_np.shape[:2]
        if w == tw and h == th:
            return img_np
        pil_img = Image.fromarray(img_np)
        pil_resized = pil_img.resize((tw, th), Image.LANCZOS)
        return np.array(pil_resized)


# ---------------------------------------------------------------------------
# 3. AI 复刻图生成（可选）
# ---------------------------------------------------------------------------

class AIReplicaGenerator:
    """调用 OpenAI 兼容图像编辑接口生成无水印复刻图。"""

    def __init__(self, api_base_url: str, api_key: str, model: str = "auto", timeout_sec: int = 90):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec

    def generate(self, original_np: np.ndarray, output_path: str, prompt: str) -> str:
        try:
            import requests
        except ImportError as e:
            raise ImportError("启用AI复刻图需要安装 requests，请先执行: pip install requests") from e

        if not self.api_base_url or not self.api_key:
            raise ValueError("AI API配置不完整，请提供 ai_api_base_url 与 ai_api_key。")
        resolved_model = self._resolve_model(requests)

        fd, tmp_png = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            Image.fromarray(original_np).save(tmp_png, format="PNG")
            endpoint = f"{self.api_base_url}/images/edits"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": resolved_model,
                "prompt": prompt,
                "response_format": "b64_json",
            }
            with open(tmp_png, "rb") as f:
                files = {"image": ("input.png", f, "image/png")}
                resp = requests.post(
                    endpoint,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=self.timeout_sec,
                )
            if resp.status_code >= 400:
                raise RuntimeError(f"AI接口请求失败: HTTP {resp.status_code} - {resp.text[:300]}")
            payload = resp.json()
            b64 = self._extract_b64(payload)
            if not b64:
                raise RuntimeError(f"AI接口返回中未找到图像数据: {json.dumps(payload)[:300]}")
            with open(output_path, "wb") as out:
                out.write(base64.b64decode(b64))
            return output_path
        finally:
            if os.path.exists(tmp_png):
                os.remove(tmp_png)

    def _resolve_model(self, requests_module) -> str:
        """自动识别可用图像模型；失败时回退到 gpt-image-1。"""
        configured = (self.model or "").strip()
        if configured and configured.lower() not in ("auto", "automatic", "default"):
            return configured

        candidates = self._fetch_models(requests_module)
        # 优先常见图像编辑/生成模型
        preferred_patterns = ("gpt-image", "image", "vision")
        for name in candidates:
            lower = name.lower()
            if any(p in lower for p in preferred_patterns):
                print(f"[信息] 自动识别AI模型: {name}")
                return name

        if candidates:
            print(f"[信息] 未找到明显图像模型，使用首个可用模型: {candidates[0]}")
            return candidates[0]

        print("[警告] 自动识别AI模型失败，回退到 gpt-image-1")
        return "gpt-image-1"

    def _fetch_models(self, requests_module) -> List[str]:
        endpoint = f"{self.api_base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests_module.get(endpoint, headers=headers, timeout=min(self.timeout_sec, 30))
            if resp.status_code >= 400:
                return []
            payload = resp.json()
            data = payload.get("data", []) if isinstance(payload, dict) else []
            names: List[str] = []
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    names.append(item["id"])
            return names
        except Exception:
            return []

    @staticmethod
    def _extract_b64(payload: dict) -> Optional[str]:
        if isinstance(payload, dict):
            if "image_base64" in payload and isinstance(payload["image_base64"], str):
                return payload["image_base64"]
            if "data" in payload and isinstance(payload["data"], list) and payload["data"]:
                first = payload["data"][0]
                if isinstance(first, dict):
                    if isinstance(first.get("b64_json"), str):
                        return first["b64_json"]
                    if isinstance(first.get("image_base64"), str):
                        return first["image_base64"]
        return None


def _scale_by_ref(value: int, current: int, ref: int, min_v: int, max_v: int) -> int:
    scaled = int(round(value * (current / max(ref, 1))))
    return max(min_v, min(max_v, scaled))


# ---------------------------------------------------------------------------
# 4. 完整处理流水线
# ---------------------------------------------------------------------------

def process_image(
        original_path: str,
        replica_path: Optional[str] = None,
        output_path: str = "output.png",
        visualize: bool = False,
        # 自动检测
        auto_detect: bool = False,
        ai_replicate: bool = False,
        ai_api_base_url: str = "",
        ai_api_key: str = "",
        ai_model: str = "auto",
        ai_prompt: str = "保留主体构图和风格，完整去除所有底部水印、角标与文字标识，输出自然无痕迹图像。",
        # 底部水印参数（auto_detect=False 时生效）
        strip_height: int = 90,
        blend_zone: int = 20,
        # 左下角小水印修复参数
        corner_inpaint_x1: int = 8,
        corner_inpaint_x2: int = 125,
        corner_inpaint_y1_offset: int = 58,
        corner_inpaint_y2_offset: int = 10,
        corner_tex_offset_y: int = -60,
        corner_tex_blur: float = 2.0,
        scale_params: bool = True,
) -> str:
    """
    完整处理入口。

    Args:
        original_path: 原始有水印图像路径
        replica_path:  AI复刻清洁图像路径（可选）
        output_path:   输出路径
        visualize:     是否弹出matplotlib预览
        auto_detect:   是否自动检测水印位置（True时自动覆盖strip_height/blend_zone）
        其余参数为水印区域调优参数，默认适配1376×768样例图。

    Returns:
        输出文件的绝对路径
    """
    # 1. 解码图像
    print(f"[1/5] 读取原图: {original_path}")
    orig_img = ImageDecoder.load(original_path)
    orig_np = np.array(orig_img)
    h, w = orig_np.shape[:2]
    print(f"      尺寸: {w}×{h}")

    replica_np = None
    ai_replica_tmp = None
    if replica_path:
        print(f"[2/5] 读取复刻图: {replica_path}")
        rep_img = ImageDecoder.load(replica_path)
        replica_np = np.array(rep_img.convert("RGB"))
        print(f"      尺寸: {replica_np.shape[1]}×{replica_np.shape[0]}")
    elif ai_replicate:
        print("[2/5] 未提供复刻图，正在调用AI生成复刻图...")
        ai_replica_tmp = _temp_path(output_path, "_ai_replica.png")
        generator = AIReplicaGenerator(
            api_base_url=ai_api_base_url,
            api_key=ai_api_key,
            model=ai_model,
        )
        generator.generate(orig_np, ai_replica_tmp, ai_prompt)
        rep_img = ImageDecoder.load(ai_replica_tmp)
        replica_np = np.array(rep_img.convert("RGB"))
        print(f"      AI复刻图已生成: {ai_replica_tmp}")
    else:
        print("[2/5] 未提供复刻图，将仅使用Inpainting模式。")

    # ---- 自动检测水印 ----
    detected_params = None
    if auto_detect:
        print("[2.5/5] 自动检测水印位置...")
        try:
            from watermark_detector import WatermarkDetector
            detector = WatermarkDetector(use_ocr=False)
            mask, meta = detector.detect(orig_np)
            detected_params = {
                "detected": meta.detected,
                "confidence": meta.confidence,
                "strip_height": meta.suggested_strip_height,
                "blend_zone": meta.suggested_blend_zone,
                "method": meta.method,
            }
            if meta.detected:
                strip_height = meta.suggested_strip_height
                blend_zone = meta.suggested_blend_zone
                print(f"      检测到水印 (置信度 {meta.confidence:.2f}, 方法: {meta.method})")
                print(f"      自动参数: strip_height={strip_height}, blend_zone={blend_zone}")
            else:
                print(f"      未检测到水印 (置信度 {meta.confidence:.2f})")
        except Exception as e:
            print(f"      [警告] 自动检测失败: {e}，使用默认参数")

    if scale_params and not auto_detect:
        strip_height = _scale_by_ref(strip_height, h, 768, 36, max(60, h // 2))
        blend_zone = _scale_by_ref(blend_zone, h, 768, 8, 80)
        corner_inpaint_x1 = _scale_by_ref(corner_inpaint_x1, w, 1376, 2, max(20, w // 4))
        corner_inpaint_x2 = _scale_by_ref(corner_inpaint_x2, w, 1376, 30, max(120, w // 2))
        corner_inpaint_y1_offset = _scale_by_ref(corner_inpaint_y1_offset, h, 768, 24, max(80, h // 2))
        corner_inpaint_y2_offset = _scale_by_ref(corner_inpaint_y2_offset, h, 768, 6, 40)
        corner_tex_offset_y = -_scale_by_ref(abs(corner_tex_offset_y), h, 768, 16, 120)

    # 2. 初始化引擎
    engine = WatermarkRemover(orig_np, replica_np)
    working = orig_np.copy()

    # 3. 粗去除：底部水印带替换
    print(f"[3/5] 去除底部水印带 (高度={strip_height}px, 过渡={blend_zone}px)...")
    working = engine.remove_bottom_strip(strip_height=strip_height, blend_zone=blend_zone)

    # 4. 精修复：左下角小水印（如AI角标或残留）
    print("[4/5] 精修左下角残留水印...")
    y1 = h - corner_inpaint_y1_offset
    y2 = h - corner_inpaint_y2_offset

    # 先对左下角做OpenCV Inpainting
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, corner_inpaint_x1:corner_inpaint_x2] = 255
    working = WatermarkRemover.inpaint_region(working, mask, radius=3)

    # 再用纹理合成对Inpainting边缘做平滑补充
    working = WatermarkRemover.texture_synthesis_fill(
        working,
        y_top=max(0, h - 75),
        y_bottom=h,
        x_left=0,
        x_right=180,
        source_offset_y=corner_tex_offset_y,
        source_offset_x=10,
        blur_sigma=corner_tex_blur,
    )

    # 5. 保存
    print(f"[5/5] 保存结果: {output_path}")
    out = Image.fromarray(working)
    out.save(output_path, quality=95)

    # 额外保存JPG版本（若输出路径是png，同时存同名jpg）
    if output_path.lower().endswith(".png"):
        jpg_path = output_path[:-4] + ".jpg"
        out.save(jpg_path, quality=95)
        print(f"      同时保存JPG: {jpg_path}")

    # 6. 可视化（调试用）
    if visualize:
        _show_comparison(orig_np, working)

    abs_output = str(Path(output_path).resolve())
    if ai_replica_tmp and os.path.exists(ai_replica_tmp):
        os.remove(ai_replica_tmp)
    return abs_output


# ---------------------------------------------------------------------------
# 5. 工具函数
# ---------------------------------------------------------------------------

def _temp_path(src: str, suffix: str) -> str:
    """生成同级目录的临时文件路径。"""
    p = Path(src)
    return str(p.parent / (p.stem + "_tmp" + suffix))


def _show_comparison(original: np.ndarray, cleaned: np.ndarray):
    """使用matplotlib显示处理前后对比。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[警告] 未安装matplotlib，跳过可视化。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].imshow(original)
    axes[0].set_title("Original (with watermark)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(cleaned)
    axes[1].set_title("Cleaned (watermark removed)", fontsize=12)
    axes[1].axis("off")

    plt.suptitle("Watermark Removal Result", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 底部区域放大
    h = original.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(18, 3))
    axes[0].imshow(original[h - 100:, :])
    axes[0].set_title("Original Bottom")
    axes[0].axis("off")

    axes[1].imshow(cleaned[h - 100:, :])
    axes[1].set_title("Cleaned Bottom")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. 命令行入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="智能水印去除工具 - 基于AI复刻图与OpenCV Inpainting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 推荐：原图 + AI复刻图（效果最佳）
  python watermark_remover.py --original photo.heic --replica ai_copy.png --output clean.png

  # 降级：仅原图（适合小水印，大水印效果有限）
  python watermark_remover.py --original photo.jpg --output clean.png

  # 开启调试预览
  python watermark_remover.py -i input.png -r replica.png -o out.png --visualize
        """
    )
    parser.add_argument("-i", "--original", required=True, help="原始有水印图像路径")
    parser.add_argument("-r", "--replica", default=None, help="AI复刻清洁图像路径（可选）")
    parser.add_argument("-o", "--output", default="output.png", help="输出图像路径（默认: output.png）")
    parser.add_argument("--visualize", action="store_true", help="显示处理前后对比图")
    parser.add_argument("--auto-detect", action="store_true", help="自动检测水印位置（覆盖手动参数）")
    parser.add_argument("--ai-replicate", action="store_true", help="未提供复刻图时，自动调用AI生成复刻图")
    parser.add_argument("--ai-api-base-url", default=os.getenv("AI_API_BASE_URL", ""), help="AI接口基础地址，如 https://api.openai.com/v1")
    parser.add_argument("--ai-api-key", default=os.getenv("AI_API_KEY", ""), help="AI接口密钥，可用环境变量 AI_API_KEY")
    parser.add_argument("--ai-model", default=os.getenv("AI_MODEL", "auto"), help="AI图像模型名，默认auto自动识别")
    parser.add_argument("--ai-prompt", default="保留主体构图和风格，完整去除所有底部水印、角标与文字标识，输出自然无痕迹图像。", help="AI复刻图生成提示词")

    # 高级参数（通常保持默认，--auto-detect 时自动覆盖）
    parser.add_argument("--strip-height", type=int, default=90, help="底部替换带高度（默认90）")
    parser.add_argument("--blend-zone", type=int, default=20, help="顶部渐变过渡像素（默认20）")
    parser.add_argument("--corner-blur", type=float, default=2.0, help="左下角平滑sigma（默认2.0）")

    args = parser.parse_args()

    result_path = process_image(
        original_path=args.original,
        replica_path=args.replica,
        output_path=args.output,
        visualize=args.visualize,
        auto_detect=args.auto_detect,
        ai_replicate=args.ai_replicate,
        ai_api_base_url=args.ai_api_base_url,
        ai_api_key=args.ai_api_key,
        ai_model=args.ai_model,
        ai_prompt=args.ai_prompt,
        strip_height=args.strip_height,
        blend_zone=args.blend_zone,
        corner_tex_blur=args.corner_blur,
    )
    print(f"\n[完成] 结果已保存: {result_path}")


if __name__ == "__main__":
    main()
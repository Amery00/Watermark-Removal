#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watermark Auto-Detector
=======================
基于多策略的水印自动检测模块。

策略优先级（速度优先）：
1. 亮度/颜色异常分析（纯OpenCV，毫秒级）
2. OCR 文字检测（可选，较慢但精准）
3. Logo 几何特征检测（纯OpenCV）

Usage:
    from watermark_detector import WatermarkDetector

    detector = WatermarkDetector(use_ocr=False)  # 纯CV模式，最快
    mask, meta = detector.detect(image_array)
    # mask: uint8 二值图 (H, W), 255=水印区域
    # meta: dict 包含检测详情与建议参数
"""

import re
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class WatermarkMeta:
    """水印检测结果元数据"""
    detected: bool  # 是否检测到水印
    method: str  # 主要检测方法
    confidence: float  # 置信度 0.0-1.0
    regions: List[Tuple[int, int, int, int]]  # [(x1, y1, x2, y2), ...]
    suggested_strip_height: int  # 建议底部替换高度
    suggested_blend_zone: int  # 建议过渡区
    text_found: List[str]  # 检测到的文字内容
    logo_matched: Optional[str]  # 匹配到的logo名称

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 主检测器
# ---------------------------------------------------------------------------

class WatermarkDetector:
    """
    水印自动检测器。

    默认使用纯OpenCV方案（毫秒级速度），OCR为可选增强。
    """

    WATERMARK_KEYWORDS = [
        "抖音", "douyin", "tiktok", "小红书", "xiaohongshu",
        "微博", "weibo", "bilibili", "b站", "快手", "kuaishou",
        "知乎", "zhihu", "微信", "wechat", "公众号",
        "generated", "ai生成", "ai generated", "dalle", "midjourney",
        "stable diffusion", "sd", "dream", "artbreeder",
    ]

    WATERMARK_PATTERNS = [
        re.compile(r"@[\w\u4e00-\u9fff]+", re.IGNORECASE),
        re.compile(r"抖音号[：:]?[\w]+", re.IGNORECASE),
        re.compile(r"id[：:]?[\w]+", re.IGNORECASE),
        re.compile(r"generated\s*by\s*\w+", re.IGNORECASE),
        re.compile(r"ai\s*generated", re.IGNORECASE),
    ]

    def __init__(self, use_ocr: bool = False, focus_region: str = "bottom"):
        """
        Args:
            use_ocr: 是否启用OCR（慢但更精准）。默认False仅用CV。
            focus_region: 关注区域，"bottom"=底部25%, "corner"=左下角, "full"=全图
        """
        self.use_ocr = use_ocr
        self.focus_region = focus_region
        self._ocr_reader = None
        self._ocr_initialized = False

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def detect(
            self,
            image: Union[np.ndarray, Image.Image, str],
            min_confidence: float = 0.25,
    ) -> Tuple[np.ndarray, WatermarkMeta]:
        """
        检测图像中的水印区域。

        Returns:
            (mask, meta): mask为uint8二值图(255=水印), meta为检测详情
        """
        arr = self._to_array(image)
        h, w = arr.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        regions: List[Tuple[int, int, int, int]] = []
        texts_found: List[str] = []
        methods_used: List[str] = []
        confidence_scores: List[float] = []
        logo_match: Optional[str] = None

        # === 策略1: 亮度/颜色异常分析（默认启用，最快） ===
        cv_regions, cv_conf = self._detect_by_color_analysis(arr)
        if cv_regions:
            regions.extend(cv_regions)
            confidence_scores.append(cv_conf)
            methods_used.append("color_analysis")

        # === 策略2: Logo 几何特征检测 ===
        logo_regions, logo_name, logo_conf = self._detect_by_logo_geometry(arr)
        if logo_regions:
            regions.extend(logo_regions)
            confidence_scores.append(logo_conf)
            methods_used.append("logo_geo")
            logo_match = logo_name

        # === 策略3: OCR（可选，仅当用户明确要求时） ===
        if self.use_ocr:
            ocr_regions, ocr_texts, ocr_conf = self._detect_by_ocr(arr)
            if ocr_regions:
                regions.extend(ocr_regions)
                texts_found.extend(ocr_texts)
                confidence_scores.append(ocr_conf)
                methods_used.append("ocr")

        # 合并 mask
        for (x1, y1, x2, y2) in regions:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

        # 形态学膨胀使 mask 更完整
        if mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)

        confidence = max(confidence_scores) if confidence_scores else 0.0
        detected = confidence >= min_confidence and mask.any()

        # 估算参数
        strip_h, blend_z = self._estimate_params(mask, h, w)

        meta = WatermarkMeta(
            detected=detected,
            method="+".join(methods_used) if methods_used else "none",
            confidence=round(min(confidence, 1.0), 3),
            regions=regions,
            suggested_strip_height=strip_h,
            suggested_blend_zone=blend_z,
            text_found=texts_found,
            logo_matched=logo_match,
        )
        return mask, meta

    def detect_params(self, image: Union[np.ndarray, Image.Image, str]) -> Dict:
        """快速检测，仅返回建议参数"""
        _, meta = self.detect(image)
        return {
            "detected": meta.detected,
            "confidence": meta.confidence,
            "strip_height": meta.suggested_strip_height,
            "blend_zone": meta.suggested_blend_zone,
            "texts": meta.text_found,
            "logo": meta.logo_matched,
            "method": meta.method,
        }

    # ==================================================================
    # 检测策略实现
    # ==================================================================

    def _detect_by_color_analysis(self, arr: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], float]:
        """纯CV亮度异常检测 — 毫秒级"""
        h, w = arr.shape[:2]
        regions: List[Tuple[int, int, int, int]] = []

        # 底部关注区
        roi_y_start = int(h * 0.65)
        roi = arr[roi_y_start:, :]
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 30:
            return [], 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # 背景亮度（大核模糊近似）
        bg = cv2.GaussianBlur(gray, (61, 61), 0)
        diff = cv2.subtract(gray, bg)

        # 显著亮于背景的区域
        _, bright_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # 绝对高亮（接近白色文字）
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 交集：既相对亮又绝对亮
        combined = cv2.bitwise_and(bright_mask, white_mask)

        # 形态学
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # 连通域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
        total_area = 0
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < 60:
                continue
            if bh < 6 or bw < 15:
                continue
            aspect = bw / max(bh, 1)
            # 水印文字通常是横向条带
            if aspect < 0.8 and area < 400:
                continue

            regions.append((x, y + roi_y_start, x + bw, y + bh + roi_y_start))
            total_area += area

        roi_area = roi_h * roi_w
        confidence = min(total_area / max(roi_area * 0.03, 1), 1.0)
        return regions, confidence

    def _detect_by_logo_geometry(self, arr: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], Optional[str], float]:
        """检测底部左侧的圆形/方形 Logo（纯CV）"""
        h, w = arr.shape[:2]
        regions = []

        # 左下角区域
        corner = arr[int(h * 0.75):, :int(w * 0.25)]
        ch, cw = corner.shape[:2]
        if ch < 20 or cw < 30:
            return [], None, 0.0

        gray = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (13, 13), 0)

        # 找亮区轮廓
        _, thresh = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        confidence = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150 or area > 8000:
                continue
            x, y, cw_r, ch_r = cv2.boundingRect(cnt)
            aspect_ratio = cw_r / max(ch_r, 1)
            # 接近圆形或方形（Logo常见形状）
            if 0.5 < aspect_ratio < 2.0:
                # 位于左下角
                cx = x + cw_r // 2
                cy = y + ch_r // 2
                if cx < cw * 0.4 and cy > ch * 0.5:
                    regions.append((x, y + int(h * 0.75), x + cw_r, y + ch_r + int(h * 0.75)))
                    confidence = max(confidence, 0.4)

        return regions, "brand_logo" if regions else None, confidence

    def _detect_by_ocr(self, arr: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[str], float]:
        """OCR文字检测 — 较慢但精准"""
        h, w = arr.shape[:2]
        regions = []
        texts = []
        confidence = 0.0

        # ROI: 底部25%
        roi = arr[int(h * 0.75):, :]
        offset_y = int(h * 0.75)

        ocr_results = self._run_easyocr(roi) or self._run_tesseract(roi)
        if not ocr_results:
            return [], [], 0.0

        for item in ocr_results:
            if isinstance(item, tuple) and len(item) == 3:
                bbox, text, conf = item
                text = text.strip().lower()
            else:
                continue

            is_watermark = self._is_watermark_text(text)
            if is_watermark or conf > 0.7:
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))
                else:
                    continue

                pad_x, pad_y = 20, 8
                regions.append((
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y) + offset_y,
                    min(w, x2 + pad_x),
                    min(h, y2 + pad_y) + offset_y,
                ))
                texts.append(text)
                confidence = max(confidence, conf)

        return regions, texts, confidence

    def _is_watermark_text(self, text: str) -> bool:
        text_lower = text.lower().strip()
        for kw in self.WATERMARK_KEYWORDS:
            if kw.lower() in text_lower:
                return True
        for pat in self.WATERMARK_PATTERNS:
            if pat.search(text):
                return True
        return False

    def _run_easyocr(self, roi: np.ndarray):
        if self._ocr_reader is None and not self._ocr_initialized:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
            except Exception:
                self._ocr_initialized = True  # 标记已尝试，不再重试
                return None
        if self._ocr_reader is None:
            return None
        try:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if roi.shape[2] == 3 else roi
            return self._ocr_reader.readtext(rgb)
        except Exception:
            return None

    def _run_tesseract(self, roi: np.ndarray):
        try:
            import pytesseract
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            results = []
            for i, text in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if conf > 30 and text.strip():
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    results.append((bbox, text, conf / 100.0))
            return results
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _estimate_params(self, mask: np.ndarray, h: int, w: int) -> Tuple[int, int]:
        if not mask.any():
            return 60, 15
        ys = np.where(mask > 0)[0]
        if len(ys) == 0:
            return 60, 15
        y_min = ys.min()
        strip_height = min(h - y_min + 20, h // 3)
        strip_height = max(strip_height, 40)
        blend_zone = max(int(strip_height * 0.25), 10)
        blend_zone = min(blend_zone, 50)
        return strip_height, blend_zone

    @staticmethod
    def _to_array(image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            return np.array(img)
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            return np.array(image)
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return image
        raise TypeError(f"不支持的图像类型: {type(image)}")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def auto_detect_watermark(image_path: str, use_ocr: bool = False) -> Dict:
    """便捷函数：对单张图片执行水印检测。确保返回值为标准 Python 类型。"""
    try:
        detector = WatermarkDetector(use_ocr=use_ocr)
        mask, meta = detector.detect(image_path)
        # 强制转换为原生 Python 类型（避免 numpy bool_/float64 导致 JSON 序列化失败）
        return {
            "success": True,
            "detected": bool(meta.detected),
            "confidence": float(meta.confidence),
            "suggested_params": {
                "strip_height": int(meta.suggested_strip_height),
                "blend_zone": int(meta.suggested_blend_zone),
            },
            "texts": list(meta.text_found),
            "logo": str(meta.logo_matched) if meta.logo_matched else None,
            "method": str(meta.method),
            "message": f"检测到水印 (置信度 {float(meta.confidence):.2f})" if meta.detected else "未检测到明显水印",
        }
    except Exception as e:
        return {
            "success": False,
            "detected": False,
            "confidence": 0.0,
            "message": f"检测失败: {str(e)}",
        }


# ---------------------------------------------------------------------------
# 命令行测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import argparse
    import time

    parser = argparse.ArgumentParser(description="水印自动检测")
    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("--mask-output", help="输出 mask 图像路径")
    parser.add_argument("--use-ocr", action="store_true", help="启用OCR（更慢但更精准）")
    args = parser.parse_args()

    print(f"正在检测: {args.image}")
    t0 = time.time()
    detector = WatermarkDetector(use_ocr=args.use_ocr)
    mask, meta = detector.detect(args.image)
    elapsed = (time.time() - t0) * 1000

    print(f"\n检测结果 ({elapsed:.1f}ms):")
    print(f"  检测到水印: {meta.detected}")
    print(f"  置信度: {meta.confidence}")
    print(f"  检测方法: {meta.method}")
    print(f"  建议参数: strip_height={meta.suggested_strip_height}, blend_zone={meta.suggested_blend_zone}")
    print(f"  检测文字: {meta.text_found}")
    print(f"  Logo匹配: {meta.logo_matched}")
    print(f"  区域数: {len(meta.regions)}")

    if args.mask_output:
        cv2.imwrite(args.mask_output, mask)
        print(f"\nMask 已保存: {args.mask_output}")

    print(f"\nJSON:")
    print(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2))
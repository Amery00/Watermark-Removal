#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watermark Remover GUI
=====================
水印去除工具的可视化桌面应用。

基于 tkinter + PIL 构建，零额外GUI依赖。
支持：原图预览、AI复刻图融合、参数实时调整、处理前后对比、结果保存、
      水印自动检测、API服务启动。

Usage:
    python watermark_gui.py
"""

import os
import sys
import threading
import io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# 导入核心处理模块（确保 watermark_remover.py 在同目录）
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from watermark_remover import process_image, ImageDecoder, WatermarkRemover
    from watermark_detector import WatermarkDetector, auto_detect_watermark
except ImportError as e:
    print(f"[错误] 无法导入核心模块: {e}")
    print("      请确保 watermark_remover.py / watermark_detector.py 与本文件在同一目录。")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 颜色主题
# ---------------------------------------------------------------------------
THEME = {
    "bg": "#f5f6fa",
    "panel_bg": "#ffffff",
    "primary": "#4a69bd",
    "primary_hover": "#3c5aa6",
    "accent": "#6a89cc",
    "text": "#2f3640",
    "text_light": "#718093",
    "border": "#dcdde1",
    "success": "#27ae60",
    "warning": "#e67e22",
    "danger": "#c0392b",
    "log_bg": "#f8f9fa",
}


# ---------------------------------------------------------------------------
# GUI 主类
# ---------------------------------------------------------------------------

class WatermarkGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("水印去除工具 - Watermark Remover")
        self.geometry("1320x860")
        self.minsize(1020, 640)
        self.configure(bg=THEME["bg"])

        # 内部状态
        self.original_path: Optional[str] = None
        self.replica_path: Optional[str] = None
        self.result_array: Optional[np.ndarray] = None
        self.result_path: Optional[str] = None
        self.original_array: Optional[np.ndarray] = None
        self.replica_array: Optional[np.ndarray] = None
        self.auto_detected_params: Optional[dict] = None
        self.api_process = None

        self.preview_mode = tk.StringVar(value="original")
        self.auto_mode = tk.BooleanVar(value=False)
        self.use_ocr = tk.BooleanVar(value=False)
        self.ai_replicate = tk.BooleanVar(value=False)

        self._build_styles()
        self._build_header()
        self._build_body()
        self._build_status()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Action.TButton", font=("Microsoft YaHei", 11, "bold"), foreground="white", background=THEME["primary"], padding=(14, 7), relief="flat")
        style.map("Action.TButton", background=[("active", THEME["primary_hover"]), ("pressed", THEME["primary"])])
        style.configure("Secondary.TButton", font=("Microsoft YaHei", 10), foreground=THEME["primary"], background=THEME["panel_bg"], padding=(8, 4))
        style.configure("Title.TLabel", font=("Microsoft YaHei", 14, "bold"), foreground=THEME["primary"], background=THEME["bg"])
        style.configure("SubTitle.TLabel", font=("Microsoft YaHei", 11, "bold"), foreground=THEME["text"], background=THEME["panel_bg"])
        style.configure("Body.TLabel", font=("Microsoft YaHei", 10), foreground=THEME["text"], background=THEME["panel_bg"])
        style.configure("Muted.TLabel", font=("Microsoft YaHei", 9), foreground=THEME["text_light"], background=THEME["panel_bg"])
        style.configure("TScale", background=THEME["panel_bg"], troughcolor=THEME["border"])
        style.configure("Horizontal.TProgressbar", background=THEME["primary"], troughcolor=THEME["border"], thickness=6)
        style.configure("TCheckbutton", background=THEME["panel_bg"], font=("Microsoft YaHei", 10))

    def _build_header(self):
        header = tk.Frame(self, bg=THEME["primary"], height=56)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Label(header, text="  Watermark Remover", font=("Microsoft YaHei", 15, "bold"), bg=THEME["primary"], fg="white").pack(side=tk.LEFT, padx=20, pady=8)
        tk.Label(header, text="智能水印去除与图像复刻工具", font=("Microsoft YaHei", 10), bg=THEME["primary"], fg="#dbeafe").pack(side=tk.LEFT, padx=4, pady=8)

    def _build_body(self):
        body = tk.Frame(self, bg=THEME["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)
        self._build_left_panel(body)
        self._build_preview_area(body)

    def _build_left_panel(self, parent):
        panel = tk.Frame(parent, bg=THEME["panel_bg"], width=360, bd=1, relief="solid", highlightbackground=THEME["border"])
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        panel.pack_propagate(False)

        # --- 文件选择 ---
        ttk.Label(panel, text="文件选择", style="SubTitle.TLabel").pack(anchor=tk.W, padx=16, pady=(16, 8))
        self._build_file_row(panel, "原图 (有水印)", "original", self._on_select_original)
        self._build_file_row(panel, "复刻图 (无水印, 可选)", "replica", self._on_select_replica)

        ttk.Separator(panel, orient="horizontal").pack(fill=tk.X, padx=16, pady=10)

        # --- 模式切换 ---
        ttk.Label(panel, text="处理模式", style="SubTitle.TLabel").pack(anchor=tk.W, padx=16, pady=(0, 8))
        mode_frame = tk.Frame(panel, bg=THEME["panel_bg"])
        mode_frame.pack(fill=tk.X, padx=16, pady=4)

        tk.Checkbutton(
            mode_frame, text="自动检测水印 (推荐)", variable=self.auto_mode,
            bg=THEME["panel_bg"], fg=THEME["text"], font=("Microsoft YaHei", 10),
            activebackground=THEME["panel_bg"], command=self._on_auto_mode_change,
        ).pack(anchor=tk.W)

        self.ocr_check = tk.Checkbutton(
            mode_frame, text="启用OCR增强 (更精准但更慢)", variable=self.use_ocr,
            bg=THEME["panel_bg"], fg=THEME["text_light"], font=("Microsoft YaHei", 9),
            activebackground=THEME["panel_bg"], state="disabled",
        )
        self.ocr_check.pack(anchor=tk.W, padx=(20, 0))

        tk.Checkbutton(
            mode_frame, text="未提供复刻图时，自动调用AI生成复刻图",
            variable=self.ai_replicate,
            bg=THEME["panel_bg"], fg=THEME["text_light"], font=("Microsoft YaHei", 9),
            activebackground=THEME["panel_bg"],
        ).pack(anchor=tk.W, padx=(20, 0))

        self.var_ai_api_base = tk.StringVar(value=os.getenv("AI_API_BASE_URL", ""))
        self.var_ai_api_key = tk.StringVar(value=os.getenv("AI_API_KEY", ""))
        self.var_ai_model = tk.StringVar(value=os.getenv("AI_MODEL", "auto"))
        self._build_entry(panel, "AI API Base URL", self.var_ai_api_base)
        self._build_entry(panel, "AI API Key", self.var_ai_api_key, show="*")
        self._build_entry(panel, "AI Model (auto=自动识别)", self.var_ai_model)

        # 自动检测结果展示区
        self.auto_result_frame = tk.Frame(panel, bg="#fff9e6", bd=1, relief="solid", highlightbackground=THEME["warning"])
        self.auto_result_frame.pack(fill=tk.X, padx=16, pady=8)
        self.auto_result_frame.pack_forget()  # 默认隐藏

        self.auto_detected_label = tk.Label(
            self.auto_result_frame, text="未检测",
            font=("Microsoft YaHei", 9), fg=THEME["text"], bg="#fff9e6",
            justify=tk.LEFT, wraplength=320, padx=8, pady=6,
        )
        self.auto_detected_label.pack(anchor=tk.W)

        ttk.Separator(panel, orient="horizontal").pack(fill=tk.X, padx=16, pady=10)

        # --- 参数调节 ---
        ttk.Label(panel, text="处理参数", style="SubTitle.TLabel").pack(anchor=tk.W, padx=16, pady=(0, 8))

        self._build_slider(panel, "底部替换高度 (px)", "strip_height", 50, 250, 90,
                          tooltip="从图像底部向上替换的像素高度，覆盖水印所在区域")
        self._build_slider(panel, "过渡融合区 (px)", "blend_zone", 0, 60, 20,
                          tooltip="替换区域顶部的渐变过渡带，越大越平滑")
        self._build_slider(panel, "左下角模糊度 (σ)", "corner_blur", 0.5, 5.0, 2.0, step=0.5,
                          tooltip="左下角修复区域的高斯模糊强度，消除拼接痕迹")

        # 检测按钮
        self.detect_btn = ttk.Button(
            panel, text="检测水印位置",
            style="Secondary.TButton",
            command=self._on_detect_only,
        )
        self.detect_btn.pack(fill=tk.X, padx=16, pady=(8, 0))

        ttk.Separator(panel, orient="horizontal").pack(fill=tk.X, padx=16, pady=12)

        # --- 操作按钮 ---
        ttk.Label(panel, text="操作", style="SubTitle.TLabel").pack(anchor=tk.W, padx=16, pady=(0, 8))
        btn_frame = tk.Frame(panel, bg=THEME["panel_bg"])
        btn_frame.pack(fill=tk.X, padx=16, pady=(0, 8))

        self.process_btn = ttk.Button(btn_frame, text="开始处理", style="Action.TButton", command=self._on_process)
        self.process_btn.pack(fill=tk.X, pady=(0, 8))

        self.save_btn = ttk.Button(btn_frame, text="保存结果", style="Secondary.TButton", command=self._on_save, state="disabled")
        self.save_btn.pack(fill=tk.X, pady=(0, 8))

        # API 服务按钮
        self.api_btn = ttk.Button(
            btn_frame, text="启动 API 服务",
            style="Secondary.TButton",
            command=self._on_toggle_api,
        )
        self.api_btn.pack(fill=tk.X, pady=(0, 8))
        self.api_status_label = ttk.Label(btn_frame, text="API 未启动", style="Muted.TLabel")
        self.api_status_label.pack(anchor=tk.W)

        # 说明文字
        hint = (
            "使用说明:\n"
            "1. 勾选「自动检测水印」可自动识别位置\n"
            "2. 选择原图与AI复刻图（可选）\n"
            "3. 点击「开始处理」去除水印\n"
            "4. 预览对比后保存结果\n"
            "5. 可启动API服务供其他程序调用"
        )
        tk.Label(panel, text=hint, font=("Microsoft YaHei", 9), fg=THEME["text_light"],
                 bg=THEME["panel_bg"], justify=tk.LEFT, wraplength=320).pack(anchor=tk.W, padx=16, pady=(4, 16))

    def _build_file_row(self, parent, label: str, key: str, command):
        frame = tk.Frame(parent, bg=THEME["panel_bg"])
        frame.pack(fill=tk.X, padx=16, pady=4)
        ttk.Label(frame, text=label, style="Body.TLabel").pack(anchor=tk.W)
        path_frame = tk.Frame(frame, bg=THEME["panel_bg"])
        path_frame.pack(fill=tk.X, pady=(4, 0))
        path_label = ttk.Label(path_frame, text="未选择", style="Muted.TLabel")
        path_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        setattr(self, f"{key}_label", path_label)
        ttk.Button(path_frame, text="浏览…", style="Secondary.TButton", command=command).pack(side=tk.RIGHT, padx=(8, 0))
        if key == "replica":
            ttk.Button(path_frame, text="清除", style="Secondary.TButton",
                      command=lambda: self._clear_replica()).pack(side=tk.RIGHT, padx=(4, 0))

    def _build_slider(self, parent, label: str, var_name: str, from_, to, default, step=1, tooltip=""):
        frame = tk.Frame(parent, bg=THEME["panel_bg"])
        frame.pack(fill=tk.X, padx=16, pady=5)
        top = tk.Frame(frame, bg=THEME["panel_bg"])
        top.pack(fill=tk.X)
        ttk.Label(top, text=label, style="Body.TLabel").pack(side=tk.LEFT)
        value_label = ttk.Label(top, text=str(default), style="Muted.TLabel")
        value_label.pack(side=tk.RIGHT)
        var = tk.DoubleVar(value=default) if isinstance(step, float) or isinstance(default, float) else tk.IntVar(value=default)
        setattr(self, f"var_{var_name}", var)
        setattr(self, f"lbl_{var_name}_value", value_label)
        scale = ttk.Scale(frame, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL,
                         command=lambda v, vn=var_name: self._on_slider_change(vn))
        scale.pack(fill=tk.X, pady=(4, 0))
        if tooltip:
            tk.Label(frame, text=tooltip, font=("Microsoft YaHei", 8), fg=THEME["text_light"],
                    bg=THEME["panel_bg"], wraplength=320, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))

    def _build_entry(self, parent, label: str, var: tk.StringVar, show: Optional[str] = None):
        frame = tk.Frame(parent, bg=THEME["panel_bg"])
        frame.pack(fill=tk.X, padx=16, pady=4)
        ttk.Label(frame, text=label, style="Body.TLabel").pack(anchor=tk.W)
        entry = tk.Entry(
            frame,
            textvariable=var,
            show=show,
            font=("Consolas", 9),
            bg="#fbfcfe",
            fg=THEME["text"],
            relief="solid",
            bd=1,
        )
        entry.pack(fill=tk.X, pady=(4, 0))

    def _on_slider_change(self, var_name: str):
        var = getattr(self, f"var_{var_name}")
        lbl = getattr(self, f"lbl_{var_name}_value")
        val = var.get()
        lbl.configure(text=f"{val:.1f}" if isinstance(var, tk.DoubleVar) else f"{int(val)}")

    def _on_auto_mode_change(self):
        auto = self.auto_mode.get()
        if auto:
            self.ocr_check.configure(state="normal")
            # 滑块变为只读展示
            for name in ["strip_height", "blend_zone", "corner_blur"]:
                getattr(self, f"var_{name}").set(getattr(self, f"var_{name}").get())
        else:
            self.ocr_check.configure(state="disabled")
            self.auto_result_frame.pack_forget()
            self.auto_detected_params = None

    def _build_preview_area(self, parent):
        container = tk.Frame(parent, bg=THEME["panel_bg"], bd=1, relief="solid", highlightbackground=THEME["border"])
        container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        toolbar = tk.Frame(container, bg=THEME["panel_bg"], height=40)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        toolbar.pack_propagate(False)
        tk.Label(toolbar, text="  图像预览", font=("Microsoft YaHei", 11, "bold"),
                fg=THEME["text"], bg=THEME["panel_bg"]).pack(side=tk.LEFT, padx=12, pady=6)

        self.mode_frame = tk.Frame(toolbar, bg=THEME["panel_bg"])
        self.mode_frame.pack(side=tk.RIGHT, padx=12)
        for text, mode in [("原图", "original"), ("结果", "result"), ("对比", "compare")]:
            btn = tk.Button(self.mode_frame, text=text, font=("Microsoft YaHei", 9),
                          fg=THEME["text_light"], bg=THEME["panel_bg"],
                          activebackground=THEME["primary"], activeforeground="white",
                          bd=1, relief="solid", highlightbackground=THEME["border"], padx=12, pady=2,
                          command=lambda m=mode: self._set_preview_mode(m))
            btn.pack(side=tk.LEFT, padx=2)
            setattr(self, f"btn_mode_{mode}", btn)
        self._update_mode_buttons()

        canvas_frame = tk.Frame(container, bg=THEME["panel_bg"])
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas = tk.Canvas(canvas_frame, bg="#e8e8e8", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._refresh_preview())

        self.placeholder_id = self.canvas.create_text(0, 0, text="请选择原图以开始预览",
            font=("Microsoft YaHei", 13), fill=THEME["text_light"], anchor="center")
        self._center_placeholder()
        self._build_console(container)

    def _build_console(self, parent):
        console = tk.Frame(parent, bg="#0f172a", height=180)
        console.pack(fill=tk.X, side=tk.BOTTOM)
        console.pack_propagate(False)

        top = tk.Frame(console, bg="#111827", height=28)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)
        tk.Label(
            top,
            text="  控制台日志",
            bg="#111827",
            fg="#cbd5e1",
            font=("Microsoft YaHei", 9, "bold"),
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            top,
            text="清空",
            font=("Microsoft YaHei", 8),
            bg="#1f2937",
            fg="#e5e7eb",
            relief="flat",
            command=lambda: self.console_text.delete("1.0", tk.END),
        ).pack(side=tk.RIGHT, padx=8, pady=3)

        self.console_text = tk.Text(
            console,
            bg="#0b1020",
            fg="#a7f3d0",
            insertbackground="#a7f3d0",
            font=("Consolas", 9),
            relief="flat",
            bd=0,
            wrap="word",
        )
        self.console_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 6))
        self._append_log("GUI 已启动，等待操作。")

    def _append_log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"

        def _write():
            self.console_text.insert(tk.END, line)
            self.console_text.see(tk.END)

        if threading.current_thread() is threading.main_thread():
            _write()
        else:
            self.after(0, _write)

    def _center_placeholder(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.coords(self.placeholder_id, w // 2, h // 2)

    def _set_preview_mode(self, mode: str):
        self.preview_mode.set(mode)
        self._update_mode_buttons()
        self._refresh_preview()

    def _update_mode_buttons(self):
        current = self.preview_mode.get()
        for mode in ["original", "result", "compare"]:
            btn = getattr(self, f"btn_mode_{mode}", None)
            if btn is None: continue
            if mode == current:
                btn.configure(bg=THEME["primary"], fg="white", activebackground=THEME["primary_hover"])
            else:
                btn.configure(bg=THEME["panel_bg"], fg=THEME["text_light"], activebackground=THEME["primary"])

    def _build_status(self):
        status = tk.Frame(self, bg=THEME["log_bg"], height=32)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        status.pack_propagate(False)
        self.status_text = tk.Label(status, text="就绪", font=("Microsoft YaHei", 9),
                                   bg=THEME["log_bg"], fg=THEME["text_light"], anchor=tk.W)
        self.status_text.pack(side=tk.LEFT, padx=12, pady=4)
        self.progress = ttk.Progressbar(status, style="Horizontal.TProgressbar",
                                         mode="determinate", length=200)
        self.progress.pack(side=tk.RIGHT, padx=12, pady=6)
        self.progress["value"] = 0

    def _set_status(self, text: str, progress: Optional[int] = None):
        self.status_text.configure(text=text)
        if progress is not None:
            self.progress["value"] = progress
        self.update_idletasks()

    # ------------------------------------------------------------------
    # 文件选择
    # ------------------------------------------------------------------
    def _on_select_original(self):
        path = filedialog.askopenfilename(
            title="选择原图",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.heic *.heif *.bmp *.webp"),
                       ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
                       ("HEIC/HEIF", "*.heic *.heif"), ("所有文件", "*.*")])
        if not path: return
        self.original_path = path
        self.original_label.configure(text=Path(path).name)
        self._append_log(f"已选择原图: {path}")
        self._load_original_preview()
        self._set_status(f"已加载原图: {Path(path).name}")
        # 如果开启了自动模式，自动检测
        if self.auto_mode.get():
            self._on_detect_only()

    def _on_select_replica(self):
        path = filedialog.askopenfilename(
            title="选择AI复刻图",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("所有文件", "*.*")])
        if not path: return
        self.replica_path = path
        self.replica_label.configure(text=Path(path).name)
        self._append_log(f"已选择复刻图: {path}")
        self._set_status(f"已加载复刻图: {Path(path).name}")

    def _clear_replica(self):
        self.replica_path = None
        self.replica_label.configure(text="未选择")
        self.replica_array = None
        self._append_log("已清除复刻图")
        self._set_status("已清除复刻图")

    def _load_original_preview(self):
        try:
            img = ImageDecoder.load(self.original_path)
            self.original_array = np.array(img)
            self.preview_mode.set("original")
            self._update_mode_buttons()
            self._refresh_preview()
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载原图:\n{e}")
            self.original_path = None
            self.original_label.configure(text="未选择")

    # ------------------------------------------------------------------
    # 仅检测水印（不处理）
    # ------------------------------------------------------------------
    def _on_detect_only(self):
        if not self.original_path:
            messagebox.showwarning("缺少文件", "请先选择原图！")
            return
        self._set_status("正在检测水印…", progress=30)
        self._append_log(f"开始检测水印 (OCR={'开' if self.use_ocr.get() else '关'})")
        thread = threading.Thread(target=self._do_detect_only, daemon=True)
        thread.start()

    def _do_detect_only(self):
        try:
            detector = WatermarkDetector(use_ocr=self.use_ocr.get())
            mask, meta = detector.detect(self.original_path)
            self.auto_detected_params = {
                "detected": meta.detected,
                "confidence": meta.confidence,
                "strip_height": meta.suggested_strip_height,
                "blend_zone": meta.suggested_blend_zone,
                "method": meta.method,
            }
            self.after(0, self._on_detect_done, meta)
        except Exception as e:
            self._append_log(f"检测失败: {e}")
            self.after(0, lambda: messagebox.showerror("检测失败", str(e)))
            self.after(0, lambda: self._set_status("检测失败", 0))

    def _on_detect_done(self, meta):
        # 更新滑块为检测值
        self.var_strip_height.set(meta.suggested_strip_height)
        self.lbl_strip_height_value.configure(text=str(meta.suggested_strip_height))
        self.var_blend_zone.set(meta.suggested_blend_zone)
        self.lbl_blend_zone_value.configure(text=str(meta.suggested_blend_zone))

        # 显示检测面板
        self.auto_result_frame.pack(fill=tk.X, padx=16, pady=8, before=self.detect_btn)
        if meta.detected:
            color = THEME["success"] if meta.confidence > 0.6 else THEME["warning"]
            text = (
                f"检测状态: ✅ 发现水印\n"
                f"置信度: {meta.confidence:.2f}\n"
                f"检测方法: {meta.method}\n"
                f"自动参数: 高度={meta.suggested_strip_height}px, 过渡={meta.suggested_blend_zone}px"
            )
        else:
            color = THEME["text_light"]
            text = (
                f"检测状态: 未检测到明显水印\n"
                f"置信度: {meta.confidence:.2f}\n"
                f"提示: 可手动调整参数或使用复刻图"
            )
        self.auto_detected_label.configure(text=text, fg=color)
        self._append_log(f"检测完成: detected={meta.detected}, confidence={meta.confidence:.2f}, method={meta.method}")
        self._set_status(f"检测完成: {'发现水印' if meta.detected else '未检测到'}", progress=100)

    # ------------------------------------------------------------------
    # 处理图像
    # ------------------------------------------------------------------
    def _on_process(self):
        if not self.original_path:
            messagebox.showwarning("缺少文件", "请先选择原图！")
            return
        if not self.replica_path and not self.auto_mode.get():
            ok = messagebox.askyesno("缺少复刻图", "未选择AI复刻图，处理效果可能不佳。\n是否继续？")
            if not ok: return

        # 获取参数
        strip_h = int(self.var_strip_height.get())
        blend_z = int(self.var_blend_zone.get())
        corner_b = self.var_corner_blur.get()
        auto_detect = self.auto_mode.get()

        self.process_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self._append_log(
            f"开始处理: auto_detect={auto_detect}, strip_height={strip_h}, blend_zone={blend_z}, "
            f"corner_blur={corner_b}, ai_replicate={self.ai_replicate.get()}"
        )
        self._set_status("正在处理…", progress=10)

        thread = threading.Thread(target=self._do_process,
            args=(strip_h, blend_z, corner_b, auto_detect), daemon=True)
        thread.start()

    def _do_process(self, strip_h: int, blend_z: int, corner_b: float, auto_detect: bool):
        try:
            self.after(0, lambda: self._set_status("正在解码图像…", progress=20))
            orig_dir = Path(self.original_path).parent
            orig_name = Path(self.original_path).stem
            out_path = str(orig_dir / f"{orig_name}_cleaned.png")

            self.after(0, lambda: self._set_status("正在去除水印…", progress=50))
            logs = io.StringIO()
            with redirect_stdout(logs), redirect_stderr(logs):
                result = process_image(
                    original_path=self.original_path,
                    replica_path=self.replica_path,
                    output_path=out_path,
                    visualize=False,
                    auto_detect=auto_detect,
                    strip_height=strip_h,
                    blend_zone=blend_z,
                    corner_tex_blur=corner_b,
                    ai_replicate=self.ai_replicate.get(),
                    ai_api_base_url=self.var_ai_api_base.get().strip(),
                    ai_api_key=self.var_ai_api_key.get().strip(),
                    ai_model=self.var_ai_model.get().strip() or "auto",
                )
            log_text = logs.getvalue().strip()
            if log_text:
                self._append_log(log_text)
            self.result_path = result
            self.result_array = np.array(ImageDecoder.load(result))
            self.after(0, self._on_process_done)
        except Exception as e:
            self.after(0, lambda err=str(e): self._on_process_error(err))

    def _on_process_done(self):
        self._append_log(f"处理完成，输出: {self.result_path}")
        self._set_status(f"处理完成: {Path(self.result_path).name}", progress=100)
        self.process_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        self.preview_mode.set("result")
        self._update_mode_buttons()
        self._refresh_preview()
        messagebox.showinfo("完成", "水印去除完成！\n点击「对比」按钮可查看处理前后差异。")

    def _on_process_error(self, err: str):
        self._append_log(f"处理失败: {err}")
        self._set_status("处理失败", progress=0)
        self.process_btn.configure(state="normal")
        messagebox.showerror("处理错误", f"水印去除过程中发生错误:\n{err}")

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def _on_save(self):
        if not self.result_path or not Path(self.result_path).exists():
            messagebox.showwarning("无结果", "没有可保存的处理结果。")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果", defaultextension=".png",
            initialfile=Path(self.result_path).name,
            filetypes=[("PNG 图像", "*.png"), ("JPEG 图像", "*.jpg *.jpeg"), ("所有文件", "*.*")])
        if not path: return
        try:
            img = Image.open(self.result_path)
            if path.lower().endswith((".jpg", ".jpeg")):
                img = img.convert("RGB")
                img.save(path, quality=95)
            else:
                img.save(path)
            self._append_log(f"结果已保存: {path}")
            self._set_status(f"已保存: {Path(path).name}")
            messagebox.showinfo("保存成功", f"结果已保存到:\n{path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    # ------------------------------------------------------------------
    # API 服务
    # ------------------------------------------------------------------
    def _on_toggle_api(self):
        if self.api_process is not None:
            self._stop_api()
        else:
            self._start_api()

    def _start_api(self):
        try:
            import subprocess
            # 启动 API 服务进程
            cmd = [sys.executable, str(SCRIPT_DIR / "api_server.py"), "--host", "0.0.0.0", "--port", "8000"]
            self.api_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(SCRIPT_DIR), text=True,
            )
            self.api_btn.configure(text="停止 API 服务")
            self.api_status_label.configure(text="API 运行中: http://0.0.0.0:8000", foreground=THEME["success"])
            self._append_log("API 服务已启动: http://0.0.0.0:8000")
            self._set_status("API 服务已启动: http://0.0.0.0:8000")

            # 在新线程中监控进程输出
            def _monitor():
                try:
                    while self.api_process and self.api_process.poll() is None:
                        line = self.api_process.stdout.readline()
                        if line:
                            print(f"[API] {line.strip()}")
                            self._append_log(f"[API] {line.strip()}")
                except Exception:
                    pass
            threading.Thread(target=_monitor, daemon=True).start()

        except Exception as e:
            messagebox.showerror("启动失败", f"无法启动 API 服务:\n{e}")

    def _stop_api(self):
        if self.api_process:
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=3)
            except Exception:
                self.api_process.kill()
            self.api_process = None
        self.api_btn.configure(text="启动 API 服务")
        self.api_status_label.configure(text="API 未启动", foreground=THEME["text_light"])
        self._append_log("API 服务已停止")
        self._set_status("API 服务已停止")

    # ------------------------------------------------------------------
    # 预览渲染
    # ------------------------------------------------------------------
    def _refresh_preview(self):
        mode = self.preview_mode.get()
        if mode == "original":
            self._draw_single(self.original_array)
        elif mode == "result":
            self._draw_single(self.result_array)
        elif mode == "compare":
            self._draw_compare()
        else:
            self._draw_single(None)

    def _draw_single(self, arr: Optional[np.ndarray]):
        self.canvas.delete("all")
        if arr is None:
            self.placeholder_id = self.canvas.create_text(0, 0, text="无图像可预览",
                font=("Microsoft YaHei", 13), fill=THEME["text_light"], anchor="center")
            self._center_placeholder()
            return
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        h, w = arr.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        pil_img = Image.fromarray(arr)
        if scale < 1.0:
            pil_img = pil_img.resize((nw, nh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        self.canvas.image = photo
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=photo)

        # 画水印区域提示线
        if self.preview_mode.get() == "original" and self.original_array is not None:
            strip_h = int(self.var_strip_height.get())
            y_line = y + nh - int(strip_h * scale)
            self.canvas.create_line(x, y_line, x + nw, y_line, fill=THEME["danger"], width=2, dash=(6, 4))
            self.canvas.create_text(
                x + nw // 2, y_line - 8,
                text=f"水印区域 (底部 {strip_h}px)",
                fill=THEME["danger"], font=("Microsoft YaHei", 9), anchor="s",
            )

    def _draw_compare(self):
        self.canvas.delete("all")
        orig = self.original_array
        res = self.result_array
        if orig is None and res is None:
            self._draw_single(None)
            return
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        half_w = (cw - 24) // 2

        def _prepare(arr, max_w, max_h):
            if arr is None: return None, 0, 0
            h, w = arr.shape[:2]
            scale = min(max_w / w, max_h / h, 1.0)
            nw, nh = int(w * scale), int(h * scale)
            pil = Image.fromarray(arr)
            if scale < 1.0:
                pil = pil.resize((nw, nh), Image.LANCZOS)
            return ImageTk.PhotoImage(pil), nw, nh

        max_h = ch - 40
        photo1, nw1, nh1 = _prepare(orig, half_w, max_h)
        photo2, nw2, nh2 = _prepare(res, half_w, max_h)
        y1 = (ch - nh1) // 2 if photo1 else ch // 2
        y2 = (ch - nh2) // 2 if photo2 else ch // 2
        x1, x2 = 8, cw // 2 + 8

        if photo1:
            self.canvas.create_image(x1, y1, anchor=tk.NW, image=photo1)
            self.canvas._photo1 = photo1
            self.canvas.create_text(x1 + nw1 // 2, y1 + nh1 + 10, text="原图",
                fill=THEME["text_light"], font=("Microsoft YaHei", 10), anchor="n")
        else:
            self.canvas.create_text(x1 + half_w // 2, ch // 2, text="(未加载原图)",
                fill=THEME["text_light"], font=("Microsoft YaHei", 10), anchor="center")

        if photo2:
            self.canvas.create_image(x2, y2, anchor=tk.NW, image=photo2)
            self.canvas._photo2 = photo2
            self.canvas.create_text(x2 + nw2 // 2, y2 + nh2 + 10, text="结果",
                fill=THEME["success"], font=("Microsoft YaHei", 10), anchor="n")
        else:
            self.canvas.create_text(x2 + half_w // 2, ch // 2, text="(未生成结果)",
                fill=THEME["text_light"], font=("Microsoft YaHei", 10), anchor="center")

        self.canvas.create_line(cw // 2, 10, cw // 2, ch - 10, fill=THEME["border"], width=1, dash=(4, 4))

    # ------------------------------------------------------------------
    # 关闭
    # ------------------------------------------------------------------
    def _on_close(self):
        self._stop_api()
        self.destroy()
        self.quit()


def main():
    try:
        import cv2, numpy, scipy, PIL
    except ImportError as e:
        print(f"[错误] 缺少必要依赖: {e}")
        print("    pip install opencv-python numpy scipy Pillow")
        sys.exit(1)
    app = WatermarkGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
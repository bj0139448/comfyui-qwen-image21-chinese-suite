# -*- coding: utf-8 -*-
"""
加速开关 —— ComfyUI-ChineseImageSuite
=====================================

一个节点打包三层加速，模型进、模型出，接在「所有 LoRA 之后、KSampler 之前」：

  1. Comfy Kitchen INT8 注意力（不压步数，优化每步内部运算，Turing 也可用）
  2. TE Predictor 输出预测（保持原采样步数，预测后续步跳过实算）
  3. Qwen Image 2.1 官方 Prefix KV Cache（编辑流专用，device/dtype 可调）

三档中文档位：
  · 画质优先 —— 零预测加速（画质基准，出问题一键回退）
  · 均衡     —— Kitchen INT8 + TE 标准预测（1 连预）+ KV 无损
  · 速度     —— Kitchen INT8 + TE 激进预测（2 连预）+ KV int8

TE-Speed 模块按路径动态加载（.pyd 编译节点，源码不可见），
找不到时自动降级为「仅注意力 + KV」，status 输出会明确告知。
零第三方依赖：只用 ComfyUI 自带 API（set_model_optimized_attention /
model_options["transformer_options"]）与标准库 importlib。
"""

import importlib.machinery
import importlib.util
import os
import sys

# ============================================================
# TE-Speed 动态加载器（单例，按路径探测，找不到返回 None）
# ============================================================

_TE_CLASS = None
_TE_TRIED = False


def _probe_paths():
    """TE-Speed-QwenImage21/nodes.pyd 的候选位置，按优先级排列。"""
    paths = []
    # 1) 与本节点包同级的 custom_nodes 目录（同步安装时最常见）
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        paths.append(os.path.join(here, "..", "TE-Speed-QwenImage21", "nodes.pyd"))
    except Exception:
        pass
    # 2) 已知便携版位置（TE 整合包）。注意必须写成 "H:/..." 绝对形式：
    #    os.path.join("H:", "xxx") 是驱动器相对路径，abspath 会按当前目录解析出错
    for drive in ("H", "F", "E", "G"):
        paths.append(drive + ":/ComfyUITE_20xx_20260813/ComfyUI/custom_nodes/"
                     "TE-Speed-QwenImage21/nodes.pyd")
    # 3) 环境变量指定
    env = os.environ.get("COMFYUI_TE_SPEED_PYD")
    if env:
        paths.append(env)
    return paths


def _load_te_class():
    """加载并缓存 TE-Speed 的 TESpeedQwenImage21 类；不可用返回 None。"""
    global _TE_CLASS, _TE_TRIED
    if _TE_TRIED:
        return _TE_CLASS
    _TE_TRIED = True
    for p in _probe_paths():
        try:
            p = os.path.abspath(p)
            if not os.path.isfile(p):
                continue
            # 注意：C 扩展初始化函数名取 spec.name 的最后一段，
            # pyd 里定义的是 PyInit_nodes，故模块名末段必须是 "nodes"
            loader = importlib.machinery.ExtensionFileLoader("_cis_te_pkg.nodes", p)
            spec = importlib.util.spec_from_loader("_cis_te_pkg.nodes", loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            cls = getattr(mod, "TESpeedQwenImage21", None)
            if cls is not None and hasattr(cls, "patch"):
                _TE_CLASS = cls
                break
        except Exception:
            continue
    return _TE_CLASS


# ============================================================
# 档位表
# ============================================================

# 加速档位 -> (是否走TE预测, attention, step_cache, predictor_error_limit, 是否直切Kitchen)
#   画质优先 = 全关（真基准）；编辑流 = 直切 Kitchen 不经 TE（编辑链专用）；
#   均衡/速度 = 经 TE patch 同时切 Kitchen + 步预测（T2I 链专用，实测有效）
SPEED_GEARS = {
    "画质优先": (False, "default", None, 0.08, False),
    "编辑流":   (False, "kitchen_int8", None, 0.08, True),
    "均衡":     (True, "kitchen_int8", "te_predictor", 0.08, False),
    "速度":     (True, "kitchen_int8", "speed", 0.12, False),
}

# KV 缓存档位 -> (device, dtype)，对应官方 QwenImage21Cache 节点
KV_GEARS = {
    "auto+default 无损·自动":   ("auto", "default"),
    "auto+int8 半存·近无损":    ("auto", "int8"),
    "cpu+int8 内存缓存·省显存": ("cpu", "int8"),
    "off 关闭":                 ("off", "default"),
}


def apply_accel(model, 加速档位, 复用阈值, KV缓存档位, 详细日志=False):
    """核心逻辑（独立函数，便于测试）：返回 (model, status)。"""
    use_te, attention, step_cache, err_limit, kitchen_direct = SPEED_GEARS.get(加速档位, SPEED_GEARS["编辑流"])
    kv_device, kv_dtype = KV_GEARS.get(KV缓存档位, KV_GEARS["auto+default 无损·自动"])
    notes = []
    m = model.clone()

    # ---- 第三层：Qwen 2.1 官方 Prefix KV Cache ----
    if m.model_options is None:
        m.model_options = {}
    m.model_options.setdefault("transformer_options", {})["qwen_image21_cache"] = {
        "device": kv_device, "dtype": kv_dtype,
    }
    notes.append("KV=%s/%s" % (kv_device, kv_dtype))

    # ---- 第一层：直切 Kitchen INT8（不经 TE patch，编辑流专用）----
    if kitchen_direct:
        try:
            from comfy.ldm.modules import attention as _att
            fn = _att.get_attention_function("comfy_kitchen_int8", None)
            if fn is not None:
                m.set_model_optimized_attention(fn)
                notes.append("Kitchen INT8(直切·无TE)")
            else:
                notes.append("Kitchen不可用→原生attention")
        except Exception as e:
            notes.append("Kitchen直切失败→原生(%s)" % type(e).__name__)

    # ---- 第一、二层：TE Predictor + Kitchen INT8（T2I 链）----
    if kitchen_direct:
        pass  # 已直切，不走 TE
    elif not use_te:
        notes.append("未启用预测加速（画质基准）")
    else:
        te_cls = _load_te_class()
        if te_cls is None:
            notes.append("TE预测模块未找到→降级为仅KV")
        else:
            try:
                out = te_cls().patch(
                    m, attention=attention, step_cache=step_cache,
                    reuse_threshold=float(复用阈值),
                    predictor_error_limit=err_limit,
                    verbose=bool(详细日志),
                )
                if isinstance(out, tuple):
                    m = out[0]
                    te_status = out[1] if len(out) > 1 else ""
                    if te_status:
                        notes.append("TE:" + str(te_status))
                else:
                    m = out
                notes.insert(0, "Kitchen INT8 + TE预测(%s)" % (
                    "1连预" if step_cache == "te_predictor" else "2连预"))
            except Exception as e:
                notes.append("TE预测调用失败→降级为仅KV（%s）" % type(e).__name__)

    status = "【%s】%s" % (加速档位, " | ".join(notes))
    return m, status

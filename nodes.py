# -*- coding: utf-8 -*-
"""
ComfyUI-ChineseImageSuite —— 中文生图套件节点
=============================================

四个节点，全部零第三方依赖（只用 ComfyUI 自带的 torch）：

  ChineseT2IPanel    纯文生图面板：提示词+人物/场景预设+分辨率档+速度档 -> 提示词/宽/高/步数/加速强度
  ChineseMultiView   多视图提示词组装：模式下拉 + 人物提示词 -> 完整提示词 + 格数
  ChineseOutfitSwap  换装提示词组装：输入类型 + 服装描述 -> 换装提示词 + 建议denoise
  GridLineDraw       多视图分割线后处理：按格数在成图上画竖直分割带

架构说明（配合配套 JSON 使用）：
  · 纯文生图链零图片节点：ChineseT2IPanel 的宽/高/步数/加速强度直接接
    EmptyLatentImage / KSampler / viggle LoRA 的控件——界面只碰面板一个节点，
    切速度档 = viggle 强度 0/1 自动切换 + 步数自动跟着变，无需手动 bypass。
  · TextEncodeQwenImageEditPlus 的 image1-3 全 optional，无图时走空模板不崩；
    多视图生成器据此通吃文生图（LoadImage bypass）与参考编辑（接图）。
  · 换装走 TextEncodeQwenImageEditPlus + VAEEncode 原图起手（denoise<1 贴原图）。

设计原则：
  · 界面全中文，内部注册名为英文（上传节点库的规范）；
  · 经典 INPUT_TYPES 风格，兼容旧版 ComfyUI；
  · 数据与逻辑分离：提示词数据在 view_layouts.py / swap_prompts.py / presets.py，
    别人拿去扩展只改数据文件，不用动逻辑。
"""

import torch

from .view_layouts import VIEW_MODES, QUALITY_SENTENCE
from .swap_prompts import FACE_LOCK, SWAP_ONLY, SWAP_ONLY_PHOTO, PHOTO_SWAP_CORE, TRIVIEW_KEEP, DENOISE_SUGGEST
from .presets import CHARACTER_PRESETS, SCENE_PRESETS, RESOLUTION_PRESETS, SPEED_PRESETS
from .accel import SPEED_GEARS, KV_GEARS, apply_accel


# ============================================================
# 节点五：加速开关（三层加速一档切换，模型进模型出）
# ============================================================

class AccelSwitch:
    """加速开关：Kitchen INT8 + TE 预测 + Qwen KV Cache 三合一，接在所有 LoRA 之后、KSampler 之前。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL",),
                "加速档位": (list(SPEED_GEARS.keys()), {"default": "均衡",
                               "tooltip": "画质优先=零加速基准；编辑流=直切Kitchen无TE（换装/编辑链专用）；均衡=Kitchen+TE 1连预、速度=Kitchen+TE 2连预（纯生图链专用）"}),
                "复用阈值": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 3.0, "step": 0.01,
                               "tooltip": "TE 预测复用阈值，0.06 为原厂测试最佳；调大更快、画质偏差风险升高"}),
                "KV缓存档位": (list(KV_GEARS.keys()), {"default": "auto+default 无损·自动",
                               "tooltip": "Qwen 2.1 官方 Prefix KV Cache；编辑流（参考图/换装）收益最明显"}),
                "详细日志": ("BOOLEAN", {"default": False,
                               "tooltip": "在 ComfyUI 控制台打印 TE 预测的逐步决策与跳过统计"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("模型", "状态")
    FUNCTION = "apply"
    CATEGORY = "中文生图套件/ChineseImageSuite"
    DESCRIPTION = "加速开关：Kitchen INT8 注意力 + TE 输出预测 + Qwen KV Cache 三合一；接在所有 LoRA 之后、KSampler 之前"

    def apply(self, 模型, 加速档位, 复用阈值, KV缓存档位, 详细日志):
        return apply_accel(模型, 加速档位, 复用阈值, KV缓存档位, 详细日志)


# ============================================================
# 节点一：纯文生图面板
# ============================================================

class ChineseT2IPanel:
    """纯文生图总控：人物/场景预设 + 分辨率档 + 速度档，一次配齐。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "画面描述；人物预设/场景预设选中的句子会自动拼在后面"}),
                "人物预设": (list(CHARACTER_PRESETS.keys()), {"default": "空"}),
                "场景预设": (list(SCENE_PRESETS.keys()), {"default": "空"}),
                "分辨率档位": (list(RESOLUTION_PRESETS.keys()), {"default": "16:9 高清 1920x1088",
                                              "tooltip": "16:9 高清=2MP 直出清晰档；2K 档更慢且有 OOM 风险"}),
                "自定义宽度": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 16,
                                     "tooltip": "仅当选「自定义宽高」时生效；宽高须为 16 的倍数"}),
                "自定义高度": ("INT", {"default": 1088, "min": 64, "max": 8192, "step": 16}),
                "速度档位": (list(SPEED_PRESETS.keys()), {"default": "均衡 25步",
                                             "tooltip": "均衡/质量/精细自动关 viggle 保画质；快速自动开 viggle 提速（仅草稿用，细节略软有塑料感）"}),
                "自定义步数": ("INT", {"default": 25, "min": 1, "max": 200,
                                     "tooltip": "仅当选「自定义步数」时生效"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("完整提示词", "宽度", "高度", "步数", "加速强度")
    FUNCTION = "build"
    CATEGORY = "中文生图套件/ChineseImageSuite"
    DESCRIPTION = "纯文生图面板：预设+分辨率+速度一屏配齐，宽/高/步数/加速强度直连下游控件"

    def build(self, 提示词, 人物预设, 场景预设, 分辨率档位, 自定义宽度, 自定义高度, 速度档位, 自定义步数):
        # 提示词组装：用户词 + 人物锁 + 场景锁 + 画质句
        parts = [提示词,
                 CHARACTER_PRESETS.get(人物预设, ""),
                 SCENE_PRESETS.get(场景预设, ""),
                 QUALITY_SENTENCE]
        parts = [p.strip().rstrip("。") for p in parts if p and p.strip()]
        prompt = "。".join(parts) + "。"
        # 分辨率
        w, h = RESOLUTION_PRESETS.get(分辨率档位, (0, 0))
        if w == 0:
            w, h = 自定义宽度, 自定义高度
        # 步数 + viggle 强度
        steps, turbo = SPEED_PRESETS.get(速度档位, (25, 0.0))
        if steps == 0:
            steps = 自定义步数
        return (prompt, w, h, steps, turbo)


# ============================================================
# 节点二：多视图提示词组装
# ============================================================

class ChineseMultiView:
    """把「人物提示词 + 视图模式」组装成完整的多视图生成提示词。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "人物提示词": ("STRING", {"multiline": True, "default": "",
                                       "tooltip": "人物描述：性别/身形/服装等锁。视图模式自动拼在后面，这里只写人物。"}),
                "视图模式": (list(VIEW_MODES.keys()), {"default": "人物三视图"}),
                "自定义格布局句": ("STRING", {"multiline": True, "default": "",
                                        "tooltip": "仅当视图模式选「自定义」时生效，填格布局描述"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("完整提示词", "格数")
    FUNCTION = "build"
    CATEGORY = "中文生图套件/ChineseImageSuite"
    DESCRIPTION = "多视图提示词组装：人物提示词+视图模式 -> 完整提示词+格数（格数供分割线节点用）"

    def build(self, 人物提示词, 视图模式, 自定义格布局句):
        if 视图模式 == "自定义":
            grid = 自定义格布局句.strip()
            cells = 0
        else:
            grid, cells = VIEW_MODES[视图模式]

        parts = [p.strip().rstrip("。") for p in [人物提示词, grid, QUALITY_SENTENCE] if p and p.strip()]
        prompt = "。".join(parts) + "。"
        return (prompt, cells)


# ============================================================
# 节点二：换装提示词组装
# ============================================================

class ChineseOutfitSwap:
    """把「输入类型 + 换装模式 + 服装描述」组装成不换脸的换装提示词，并给建议 denoise。

    两种换装模式：
      · 文字描述 —— 衣服靠「服装描述」文字写（经典模式）
      · 照片对照片 —— 衣服来自第二张参考图（图2），走 EditPlus 的 image2 槽；
        此时不拼「新服装：」前缀，直接输出图1/图2 指代指令。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入类型": (list(DENOISE_SUGGEST.keys()), {"default": "单张人物图",
                                           "tooltip": "三视图模式会自动追加分格位置保持句"}),
                "换装模式": (["文字描述", "照片对照片"], {"default": "文字描述",
                                           "tooltip": "照片对照片=衣服来自第二张参考图，需在JSON里把图2接文本编码的图像2槽；提示词自动用官方英文范式（Replace the clothes in image 1 with the clothes in image 2），Qwen-VL 对英文指代遵循最强"}),
                "服装描述": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "文字描述模式：新服装的款式/材质/颜色；照片对照片模式：补充要求，可留空（如'只要上衣和裤子，鞋帽不算'）"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("换装提示词", "建议denoise")
    FUNCTION = "build"
    CATEGORY = "中文生图套件/ChineseImageSuite"
    DESCRIPTION = "换装提示词组装：自动拼脸锁句+唯一改动声明，输出换装提示词与建议denoise；照片对照片模式衣服取自第二张参考图"

    def build(self, 输入类型, 换装模式, 服装描述):
        parts = [FACE_LOCK, SWAP_ONLY_PHOTO if 换装模式 == "照片对照片" else SWAP_ONLY]
        if 输入类型 == "三视图":
            parts.append(TRIVIEW_KEEP)
        if 换装模式 == "照片对照片":
            parts.append(PHOTO_SWAP_CORE)
            if 服装描述.strip():
                parts.append(服装描述.strip())          # 补充要求直接接，不加前缀
        elif 服装描述.strip():
            parts.append("新服装：" + 服装描述.strip())
        parts = [p.strip().rstrip("。") for p in parts if p and p.strip()]
        prompt = "。".join(parts) + "。"
        return (prompt, DENOISE_SUGGEST[输入类型])


# ============================================================
# 节点三：多视图分割线后处理
# ============================================================

class GridLineDraw:
    """在多视图成图上按格数画竖直分割带（等分，100% 笔直，不依赖模型画线）。"""

    线色表 = {"浅灰": 0.75, "白": 1.0, "深灰": 0.25}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "格数": ("INT", {"default": 3, "min": 1, "max": 12, "step": 1,
                               "tooltip": "接 ChineseMultiView 的格数输出；1 表示不画线"}),
                "线色": (list(cls.线色表.keys()), {"default": "浅灰"}),
                "线宽": ("INT", {"default": 8, "min": 2, "max": 64, "step": 2}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像")
    FUNCTION = "draw"
    CATEGORY = "中文生图套件/ChineseImageSuite"
    DESCRIPTION = "多视图分割线后处理：按格数在成图上画等宽竖直分割带"

    def draw(self, 图像, 格数, 线色, 线宽):
        if 格数 < 2:
            return (图像,)
        out = 图像.clone()
        b, h, w, c = out.shape
        val = self.线色表[线色]
        half = 线宽 // 2
        for i in range(1, 格数):
            x = int(round(w * i / 格数))
            x0 = max(0, x - half)
            x1 = min(w, x0 + 线宽)
            out[:, :, x0:x1, :] = val
        return (out,)


# ============================================================
# 注册
# ============================================================

NODE_CLASS_MAPPINGS = {
    "ChineseT2IPanel": ChineseT2IPanel,
    "ChineseMultiView": ChineseMultiView,
    "ChineseOutfitSwap": ChineseOutfitSwap,
    "GridLineDraw": GridLineDraw,
    "AccelSwitch": AccelSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChineseT2IPanel": "纯文生图面板·ChineseImageSuite",
    "ChineseMultiView": "多视图提示词组装·ChineseImageSuite",
    "ChineseOutfitSwap": "换装提示词组装·不换脸",
    "GridLineDraw": "多视图分割线",
    "AccelSwitch": "加速开关·不掉画质",
}

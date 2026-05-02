# Whisper-Putian-Project: Phase 1

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

这是一个针对**莆田话（兴化语）**进行微调的端到端语音识别与翻译项目。Phase 1 验证了基于 Whisper-Small 在少量非标语料下的迁移学习可行性。

## 🛠 Tech Stack & Architecture

*   **Core Model:** OpenAI Whisper (Small)
*   **Infrastructure:** NVIDIA H100 (Training) / Windows Local CPU (Inference)
*   **Pipeline:** `Audio -> Log-Mel Spectrogram -> Encoder -> Decoder -> Text`
*   **Interface:** Gradio Web UI

## 📊 Phase 1 Training Metrics

| Hyperparameter | Value |
| :--- | :--- |
| **Dataset Size** | 2,627 samples (~2.5 hours) |
| **Batch Size** | 128 |
| **Learning Rate** | 1e-5 (Linear decay) |
| **Optimizer** | AdamW |
| **Training Steps** | 130 (approx. 6.3 Epochs) |
| **Eval Loss** | 0.88 (Checkpoint-130) |

## 核心计算与检查点 (Checkpoints)

在第一阶段训练中，模型在 **checkpoint-130** 处达到了较好的 `eval_loss`。

### Training Logic
通过下式计算训练强度：
$$\text{Epochs} = \frac{\text{Steps} \times \text{Batch Size}}{\text{Dataset Size}} \approx \frac{130 \times 128}{2627} \approx 6.33$$


## 推理部署 (Gradio UI)

项目集成了一个基于 Gradio 的交互式 Web 界面。

### 关键修复：
1.  **跨平台运行**：通过补齐 `preprocessor_config.json` 和 `tokenizer` 相关文件，支持在无 GPU 的 Windows 环境下运行。
2.  **安全性优化**：针对 `xxx.gradio.live` 的录音权限问题，建议在 Chrome 浏览器 HTTPS 环境下使用。

---

## 阶段性实验结论 (Post-Mortem)

*   **普通话偏好 (Mandarin Inertia)**：由于 Whisper 预训练权重的强先验性，模型倾向于将方言纠错为发音接近的普通话词汇。
*   **瓶颈分析**：
    1.  **数据量级**：2.6k 条语料尚不足以重塑模型的声学特征空间。
    2.  **文字标准缺失**：莆田话“有音无字”导致标注一致性较低，增加了模型的收敛难度。
    3.  **任务属性**：方言识别更接近 **Speech-to-Translation (ST)** 任务，而非单纯的 ASR。

---

## Phase 2 计划 (Roadmap)

- [ ] **Data Augmentation**: 启动大规模采集计划，目标语料库 **10,000+** 条。
- [ ] **Labeling Optimization**: 
    *   探索采用 **[拼音] + 意译文字** 的双重标注模式。
    *   引入“伪标签”机制，利用 Phase 1 模型辅助半自动标注。
- [ ] **Advanced Training**: 尝试在推理阶段加入 `Task Prefix` 指令，并尝试微调更大的 `Whisper-Medium` 模型。

---

## 📂 文件清单说明

*   `app.py`: Gradio 推理前端脚本。（app(1).py为linux版推理脚本）
*   `train_whisper.py`: 基于 Transformers 库的微调主程序。
*   `requirements.txt`: 项目依赖列表。
*   `assets/`: 存放运行截图及架构图。


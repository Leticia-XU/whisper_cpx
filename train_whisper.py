import os
import torch
import pandas as pd
from datasets import Dataset, Audio, DatasetDict
from transformers import (
    WhisperFeatureExtractor, 
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ================= 1. H100 性能猛药 =================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ================= 2. 路径与配置 =================
DATA_ROOT = "./"               
CLIPS_DIR = "./clips"          
MODEL_NAME = "openai/whisper-small" 

# ================= 3. 精准匹配你的 TSV 格式 =================
def load_custom_data(tsv_name):
    # 1.读取 tsv，明确指定列名映射
    df = pd.read_csv(os.path.join(DATA_ROOT, tsv_name), sep='\t')
    
    # 2.路径拼接：clips/目录1/xxx音频
    # 注意：如果你的 path 列里没有后缀，请在后面加上 + ".mp3" 或实际后缀
    df['audio'] = df['path'].apply(lambda x: os.path.join(CLIPS_DIR, x))
    
    # 检查文件是否存在（新手调试利器：如果路径拼错了，这里会直接报错提醒）
    sample_path = df['audio'].iloc[0]
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"找不到文件: {sample_path}。请检查 CLIPS_DIR 路径设置。")

    #3.构造数据集， 只抽取 Whisper 训练需要的两列
    df = df[['audio', 'sentence']]
    ds = Dataset.from_pandas(df)
    #4.核心转换：Audio 模块会自动处理 .m4a 和 .wav 的解码
    # 关键：将路径列转为真正的音频流，并重采样为 16kHz
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

print("正在准备莆田话数据集...")
train_dataset = load_custom_data("train.tsv")
test_dataset = load_custom_data("test.tsv")

print("正在初始化处理器...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="Chinese", task="transcribe")

def prepare_dataset(batch):
    # 1. 计算输入特征
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # 2. 编码目标文本
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# 对数据集进行预处理
# num_proc=4 可以利用多核 CPU 加速处理
print("正在将音频转换为特征图...")
train_dataset = train_dataset.map(prepare_dataset, num_proc=4)
test_dataset = test_dataset.map(prepare_dataset, num_proc=4)


# ================= 4. 加载模型组件 =================
# 针对莆田话，我们虽然用 Chinese 标识，但实际学习的是你的标注音频对
#processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="Chinese", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# 强制模型在训练时输出中文 Token
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="Chinese", task="transcribe")
model.config.suppress_tokens = []

# ================= 5. 数据整理器 =================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 1. 提取 input_features (map 已经生成好了，直接拿)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 2. 提取 labels (map 已经生成好了，直接拿)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 3. 掩码处理
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ================= 6. 训练参数 (H100 优化版) =================
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-putian-finetuned",
    per_device_train_batch_size=128,      # H100 80G 显存直接拉满到 64 甚至更多
    gradient_accumulation_steps=1, 
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=300,                      # 1小时数据建议跑 1000-1500 步
    save_steps=10,                      # 每 50 步就存一个档，防止白跑
    eval_steps=10,                      # 每 50 步做一次验证
    save_total_limit=3,           # 只保留最新的 3 个存档，省硬盘空间
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,                           # H100 核心加速模式
    save_strategy="steps",
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    report_to=["tensorboard"],
    dataloader_num_workers=4,             # 多线程加载数据，防止卡 CPU
    load_best_model_at_end=True,  # 训练结束自动加载表现最好的那个模型
    metric_for_best_model="eval_loss", # 以验证集损失作为评判标准
    remove_unused_columns=False, # 必须为 False
    label_names=["labels"]      # 确保标签名对应
)

# ================= 7. 启动 =================
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    #tokenizer=processor.feature_extractor,
)

print("炼丹开始，观察 nvidia-smi 确认 H100 是否满载...")
trainer.train()

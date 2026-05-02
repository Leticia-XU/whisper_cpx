import gradio as gr
from transformers import pipeline
import torch

# 1. 加载你练好的最强模型
# 建议指向那个 eval_loss 最低的 checkpoint 文件夹
model_path = "./whisper-putian-finetuned/checkpoint-130" 

print(f"正在加载莆田话模型: {model_path}...")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

def transcribe(audio):
    if audio is None:
        return "请先录音或上传音频文件！"
    
    # 执行识别
    result = pipe(audio)
    return result["text"]

# 2. 搭建网页界面
demo = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="说一句莆田话吧"),
    outputs=gr.Textbox(label="识别结果"),
    title="莆田话 AI 识别测试 (Whisper Fine-tuned)",
    description="点击录音按钮，说一句莆田话试试看。模型还在进化中，欢迎测试！"
)

# 3. 启动并开启外网分享
# 设置 share=True 会生成一个临时的 gradio.live 链接，可以直接发给家人
demo.launch(share=True)

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========== 1. 加载微调好的模型 ==========
model_dir = "./my-bert-model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# 标签映射（对应你的虚假新闻数据集：0=非真实，1=真实）
# 你可以根据自己的需求改成更友好的文字
label_map = {
    0: "❌ 这是一条虚假/误导性新闻",
    1: "✅ 这看起来是一条真实新闻"
}

# ========== 2. 定义推理函数 ==========
def predict_news(text):
    # 预处理文本
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128  # 和你训练时的 max_length 保持一致
    )

    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = logits.argmax().item()
        confidence = probabilities[0][predicted_class_id].item()

    # 返回结果
    result_label = label_map[predicted_class_id]
    return f"{result_label}\n\n置信度：{confidence:.2%}"

# ========== 3. 搭建 Gradio 网页界面 ==========
with gr.Blocks(title="虚假新闻检测器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📰 BERT 虚假新闻检测器")
    gr.Markdown("输入一条推文或新闻标题，模型会判断它是否真实。")

    with gr.Row():
        input_text = gr.Textbox(
            label="输入新闻/推文文本",
            placeholder="例如：Just happened: A huge earthquake hit California!",
            lines=3
        )

    with gr.Row():
        output_text = gr.Textbox(label="预测结果", lines=2)

    submit_btn = gr.Button("开始检测", variant="primary")
    submit_btn.click(fn=predict_news, inputs=input_text, outputs=output_text)

    # 添加几个示例，方便用户直接点
    gr.Examples(
        examples=[
            ["My last two weather pics from the storm on August 2nd. People packed up real fast after the temp dropped and winds picked up."],
            ["Lying Clinton sinking! Donald Trump singing: Let's Make America Great Again!"],
            ["Breaking: Scientists discover a cure for all diseases in a rare Amazonian plant."]
        ],
        inputs=input_text
    )

# ========== 4. 启动 Demo ==========
if __name__ == "__main__":
    demo.launch(share=False)  # share=False 只在本地跑，上线时不用改


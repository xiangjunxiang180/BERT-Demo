# BERT 虚假新闻检测器
基于 BERT 预训练模型的英文虚假新闻检测系统，支持在线 Web Demo 部署。

## 项目效果
- 验证集准确率：83.13%
- 任务类型：NLP 文本分类（二分类）
- 框架：PyTorch + HuggingFace Transformers + Gradio

## 功能
1. 新闻文本真实性自动检测
2. 在线交互式 Web Demo
3. 支持置信度输出

## 快速运行
### 1. 安装依赖
```bash
pip install -r requirements.txt

# DeepSeek-OCR WebUI

**Language:** [English](README.md) | [中文](README_zh.md)

基于 DeepSeek-OCR 模型的网页界面，支持中英文双语。

## 📋 版本更新记录

### [2025-10-24] - 增强表格识别展示与图像位置标注
- ✨ **表格识别展示**: 新增表格结构可视化功能，支持Markdown格式输出
- 🎯 **图片位置识别标注**: 实现交互式HTML坐标标注系统
  - 彩色边界框标注检测到的文本区域
  - 实时坐标图例，精确显示位置信息
  - 多格式输出支持（Markdown、HTML、标注图像）
  - 响应式设计，完美适配所有设备尺寸

## 快速开始

### 1. 下载模型

从以下任一源下载 DeepSeek-OCR 模型：
- **ModelScope**: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-OCR
- **Hugging Face**: https://huggingface.co/deepseek-ai/DeepSeek-OCR

### 2. 安装方式

#### 方式 A：本地安装

1. **安装依赖**
   ```bash
   conda create -n deepseek-ocr python=3.12.9 -y
   conda activate deepseek-ocr
   pip install -r requirements.txt
   pip install flash-attn==2.7.3 --no-build-isolation
   ```

2. **配置模型路径**
   
   编辑 `start_ocr_webui.py` 第 26 行：
   ```python
   # 修改此行为你的模型路径
   self.model_path = '/你的路径/DeepSeek-OCR'
   ```

3. **运行应用**
   ```bash
   python start_ocr_webui.py
   ```

4. **访问界面**
   
   浏览器打开：http://localhost:7860

#### 方式 B：Docker 部署

1. **准备模型目录**
   ```bash
   mkdir -p ./models
   # 将下载的 DeepSeek-OCR 模型放在 ./models/DeepSeek-OCR/ 目录下
   ```

2. **构建并运行**
   ```bash
   docker-compose up -d
   ```

3. **访问界面**
   
   浏览器打开：http://localhost:7860

### 3. 使用方法

1. 上传一张或多张图片
2. 输入 OCR 提示词（或使用预设提示词）
3. 点击"识别"按钮
4. 在结果/摘要选项卡中查看结果

## 系统要求

- Python 3.12+
- 支持 CUDA 的 GPU（推荐）
- 16GB+ GPU 显存
- 带 CUDA 支持的 PyTorch

## 功能特点

- 多图片批量处理
- 多种 OCR 提示词预设
- 双语界面（中英文）
- Docker 部署支持
- 实时处理进度

## 常用提示词

- **通用 OCR**：`Free OCR.`
- **转换为 Markdown**：`<|grounding|>Convert the document to markdown.`
- **提取表格**：`<|grounding|>Extract all tables and convert to markdown format.`

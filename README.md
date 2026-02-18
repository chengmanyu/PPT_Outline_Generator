# PPT Generator
Business presentations, classroom reports, and online courses necessitate a substantial amount of time for layout design and image creation. Current tools (such as PowerPoint and Illustrator) still demand manual adjustments. This project aims to develop an **end-to-end AI PPT outline generator**: by inputting plain text, it can automatically generate ppt outline . This will enhance efficiency and is applicable to the fields of education, marketing, and remote work. 


# PPT Outline Generator using Fine-Tuned Qwen2-1.5B

## Project Overview
This capstone project implements an AI system that automatically generates structured 5-slide PowerPoint outlines from any given topic using a fine-tuned Transformer model (Qwen2-1.5B-Instruct).

## Key Features
- Fine-tuning of Qwen2-1.5B with LoRA (efficient parameter tuning)
- 4-bit quantization for low GPU memory usage
- Custom dataset of 500 high-quality topic-outline pairs
- Training & validation loss curves
- Qualitative examples and quantitative perplexity metric

## Requirements
- It must be a Windows system.
- It is recommended that the device be equipped with an RTX3060 GPU or a higher-end GPU for optimal performance.


## How to Run
1. Utilise the Terminal to execute the "start.bat" file (which is provided).
2. Utilise a separate Terminal to execute the "start_frontend.bat" file (which is provided).
3. Access the website, input the title of the PPT, and then click the generate button.

## Project Demonstration Video
![Demo gif image](PPT_outline_Demo_video.gif)

## Results
- Validation loss decreases steadily over 2 epochs
- Generated outlines are coherent, structured, and topic-relevant
- Low perplexity indicates good language modeling performance

### Model Download (Hugging Face)

**Recommended version (full model)**：
→ [ChengManYu/qwen2-1.5b-ppt-outline-merged](https://huggingface.co/ChengManYu/qwen2-1.5b-ppt-outline-merged)

**LoRA version (can continue training)**：
→ [ChengManYu/qwen2-1.5b-ppt-outline-lora](https://huggingface.co/ChengManYu/qwen2-1.5b-ppt-outline-lora)

### 線上體驗

- **Kaggle Notebook**（Can be executed directly）：https://www.kaggle.com/code/manyucheng/ppt-outline-generator-with-fine-tuned-qwen2-1-5b
- **Gradio Demo**（Coming soon）

## Future Work
- Increase dataset size and diversity
- Add slide content generation
- Integrate with actual PowerPoint export

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==================== 步驟1：檢查 GPU 是否可用 ====================
print("檢查 CUDA 是否可用...")
if not torch.cuda.is_available():
    raise RuntimeError("沒有偵測到 GPU！請確認已安裝 CUDA 版 PyTorch 和 NVIDIA 驅動程式。")
else:
    print(f"GPU 可用！名稱: {torch.cuda.get_device_name(0)}")
    print(f"GPU 總記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.cuda.empty_cache()  # 清空快取，避免碎片

# ==================== 步驟2：設定模型 ====================
model_repo = "ChengManYu/qwen2-1.5b-ppt-outline-merged"  # 你的模型 repo

# 4-bit 量化設定（強烈推薦，對 3060 非常友好）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ==================== 步驟3：載入 Tokenizer 和 Model 到 GPU ====================
print("\n正在載入 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 避免警告

print("正在載入模型到 GPU（第一次會下載，需較久時間）...")
model = AutoModelForCausalLM.from_pretrained(
    model_repo,
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"  # 強制指定使用你的 RTX 3060（第一張 GPU）
)

# 額外保險：再次確保模型在 GPU 上
model = model.to("cuda:0")
model.eval()  # 設為評估模式

print("模型載入完成！已全數部署在你的 RTX 3060 上。")
print("提示：此時可開啟命令提示字元執行 'nvidia-smi -l 1' 監控 GPU 使用情況。")

# ==================== 步驟4：生成 PPT 大綱函數 ====================
def generate_ppt_outline(topic):
    prompt = f"Generate PPT outline for: {topic}\n### Outline:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")  # 明確移到 GPU
    
    with torch.no_grad():  # 節省記憶體
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,        # 可稍微加大，生成更完整大綱
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1     # 避免重複內容
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 Outline 之後的內容
    if "### Outline:" in generated_text:
        outline = generated_text.split("### Outline:")[-1].strip()
    else:
        outline = generated_text.strip()
    
    return outline

# ==================== 步驟5：互動測試迴圈 ====================
print("\n" + "="*60)
print("PPT 大綱生成器已就緒！開始輸入主題（輸入 exit 離開）")
print("="*60)

while True:
    topic = input("\n輸入主題: ").strip()
    if topic.lower() == "exit" or topic.lower() == "quit":
        print("再見！")
        break
    if not topic:
        print("請輸入有效主題！")
        continue
    
    print(f"\n正在為「{topic}」生成 PPT 大綱（使用你的 RTX 3060 加速中）...")
    try:
        outline = generate_ppt_outline(topic)
        print("\n=== 生成的 PPT 大綱 ===\n")
        print(outline)
        print("\n" + "-" * 60)
    except Exception as e:
        print(f"生成時發生錯誤: {e}")
        print("請檢查網路或模型是否正常。")

# 程式結束時可選清空 GPU 記憶體
torch.cuda.empty_cache()
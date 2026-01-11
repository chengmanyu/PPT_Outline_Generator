from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import asyncio

app = FastAPI()

# 允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型和tokenizer（初始化时加载，避免每次请求重复加载）
model = None
tokenizer = None

class OutlineRequest(BaseModel):
    topic: str

class OutlineResponse(BaseModel):
    topic: str
    outline: str

@app.on_event("startup")
async def startup_event():
    """启动时加载模型到 GPU"""
    global model, tokenizer
    
    print("初始化中...检查 GPU...")
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 GPU！")
    
    print(f"GPU 可用: {torch.cuda.get_device_name(0)}")
    print(f"GPU 总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.cuda.empty_cache()
    
    model_repo = "ChengManYu/qwen2-1.5b-ppt-outline-merged"
    
    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("正在加载模型到 GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    
    model = model.to("cuda:0")
    model.eval()
    print("✓ 模型加载完成！后端已就绪")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清空 GPU 内存"""
    torch.cuda.empty_cache()

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/generate", response_model=OutlineResponse)
async def generate_outline(request: OutlineRequest):
    """生成 PPT 大纲"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="主题不能为空")
    
    try:
        prompt = f"Generate PPT outline for: {request.topic}\n### Outline:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Outline:" in generated_text:
            outline = generated_text.split("### Outline:")[-1].strip()
        else:
            outline = generated_text.strip()
        
        return OutlineResponse(topic=request.topic, outline=outline)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

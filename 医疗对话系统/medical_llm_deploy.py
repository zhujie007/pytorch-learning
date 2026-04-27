import os
import time
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==================== 配置参数 ====================
MODEL_PATH = r"d:\桌面\实验6\MING-1.8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 获取GPU显存使用（MB）
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

# ==================== 模型加载 ====================
def load_model(model_path, use_4bit=False, use_8bit=False):
    """
    加载模型和分词器
    参数:
        model_path: 模型路径
        use_4bit: 是否使用4bit量化
        use_8bit: 是否使用8bit量化
    返回:
        model, tokenizer, memory_used, load_time
    """
    print("=" * 60)
    print("开始加载模型...")
    print("=" * 60)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    mem_before = get_gpu_memory_usage()
    
    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print("分词器加载完成!")
    
    
    
    start_time = time.time()
    
    # 加载模型
    if use_4bit:
        print("使用4bit量化加载模型...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    elif use_8bit:
        print("使用8bit量化加载模型...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    else:
        print("使用原始精度加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    load_time = time.time() - start_time
    mem_after = get_gpu_memory_usage()
    mem_used = mem_after - mem_before
    
    print(f"模型加载完成!")
    print(f"加载时间: {load_time:.2f} 秒")
    print(f"GPU显存使用: {mem_used:.2f} MB")
    print("=" * 60)
    
    return model, tokenizer, mem_used, load_time

# ==================== 文本生成 ====================
def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """
    生成模型回复
    参数:
        model: 加载的模型
        tokenizer: 分词器
        prompt: 用户输入
        max_new_tokens: 最大生成token数
    返回:
        response, inference_time, memory_used
    """
    # 构建医疗对话格式
    messages = [
        {"role": "system", "content": "你是一个专业的医疗助手，请用中文回答用户的医疗问题。"},
        {"role": "user", "content": prompt}
    ]
    
    # 应用对话模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        mem_before = 0
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            mem_after = 0
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(f"GPU显存不足: {e}")
    except Exception as e:
        raise RuntimeError(f"模型推理失败: {e}")
    
    inference_time = time.time() - start_time
    mem_used = mem_after - mem_before
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return response, inference_time, mem_used

# ==================== 实验1: 基础推理 ====================
def experiment_basic_inference():
    """
    实验要求1: 掌握本地大模型的调用方法
    优化指令输出"新冠肺炎的症状？"
    """
    print("\n" + "=" * 60)
    print("实验1: 基础推理 - 新冠肺炎的症状")
    print("=" * 60)
    
    # 加载模型
    model, tokenizer, model_mem, load_time = load_model(MODEL_PATH)
    
    # 测试问题
    prompt = "新冠肺炎的症状？"
    print(f"\n用户问题: {prompt}")
    print("-" * 60)
    
    # 生成回复
    response, inference_time, inference_mem = generate_response(model, tokenizer, prompt)
    
    print(f"\n模型回复:")
    print(response)
    print("-" * 60)
    print(f"推理时间: {inference_time:.2f} 秒")
    print(f"推理内存: {inference_mem:.2f} MB")
    
    # 清理内存
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model_mem, load_time, inference_time, inference_mem

# ==================== 实验2: 量化对比 ====================
def experiment_quantization_comparison(base_mem, base_load_time, base_inference_time):
    """
    实验要求2: 掌握大语言模型量化CPU或者GPU方式
    显示减少的内存和推理时间百分比
    """
    print("\n" + "=" * 60)
    print("实验2: 量化对比实验")
    print("=" * 60)
    
    prompt = "新冠肺炎的症状？"
    mem_8bit = load_time_8bit = inf_time_8bit = 0
    mem_4bit = load_time_4bit = inf_time_4bit = 0
    
    # 测试8bit量化
    print("\n--- 8bit量化测试 ---")
    try:
        model_8bit, tokenizer_8bit, mem_8bit, load_time_8bit = load_model(
            MODEL_PATH, use_8bit=True
        )
        
        response_8bit, inf_time_8bit, inf_mem_8bit = generate_response(model_8bit, tokenizer_8bit, prompt)
        
        print(f"\n8bit量化推理结果:")
        print(response_8bit)

        mem_reduction_8bit = ((base_mem - mem_8bit) / base_mem) * 100
        load_reduction_8bit = ((base_load_time - load_time_8bit) / base_load_time) * 100
        inference_reduction_8bit = ((base_inference_time - inf_time_8bit) / base_inference_time) * 100

        print(f"\n8bit量化对比:")
        print(f"  内存变化: {mem_reduction_8bit:.2f}% ({'减少' if mem_reduction_8bit > 0 else '增加'})")
        print(f"  加载时间变化: {load_reduction_8bit:.2f}% ({'减少' if load_reduction_8bit > 0 else '增加'})")
        print(f"  推理时间变化: {inference_reduction_8bit:.2f}% ({'减少' if inference_reduction_8bit > 0 else '增加'})")
        
        del model_8bit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"8bit量化测试失败: {e}")
    except Exception as e:
        print(f"8bit量化测试发生未知错误: {e}")
    
    # 测试4bit量化
    print("\n--- 4bit量化测试 ---")
    try:
        model_4bit, tokenizer_4bit, mem_4bit, load_time_4bit = load_model(
            MODEL_PATH, use_4bit=True
        )
        
        response_4bit, inf_time_4bit, inf_mem_4bit = generate_response(model_4bit, tokenizer_4bit, prompt)
        
        print(f"\n4bit量化推理结果:")
        print(response_4bit)

        mem_reduction_4bit = ((base_mem - mem_4bit) / base_mem) * 100
        load_reduction_4bit = ((base_load_time - load_time_4bit) / base_load_time) * 100
        inference_reduction_4bit = ((base_inference_time - inf_time_4bit) / base_inference_time) * 100

        print(f"\n4bit量化对比:")
        print(f"  内存变化: {mem_reduction_4bit:.2f}% ({'减少' if mem_reduction_4bit > 0 else '增加'})")
        print(f"  加载时间变化: {load_reduction_4bit:.2f}% ({'减少' if load_reduction_4bit > 0 else '增加'})")
        print(f"  推理时间变化: {inference_reduction_4bit:.2f}% ({'减少' if inference_reduction_4bit > 0 else '增加'})")
        
        del model_4bit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"4bit量化测试失败: {e}")
    except Exception as e:
        print(f"4bit量化测试发生未知错误: {e}")
    
    # 总结对比表
    print("\n" + "=" * 60)
    print("量化效果总结")
    print("=" * 60)
    print(f"{'模型版本':<15} {'内存(MB)':<12} {'加载时间(s)':<15} {'推理时间(s)':<15}")
    print("-" * 60)
    print(f"{'原始精度':<15} {base_mem:<12.2f} {base_load_time:<15.2f} {base_inference_time:<15.2f}")
    if load_time_8bit > 0:
        print(f"{'8bit量化':<15} {mem_8bit:<12.2f} {load_time_8bit:<15.2f} {inf_time_8bit:<15.2f}")
    if load_time_4bit > 0:
        print(f"{'4bit量化':<15} {mem_4bit:<12.2f} {load_time_4bit:<15.2f} {inf_time_4bit:<15.2f}")
    print("=" * 60)

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("MING-1.8B 中文医疗大模型本地部署")
    print(f"使用设备: {device}")
    print("=" * 60)
    
    base_mem, base_load_time, base_inf_time, base_inf_mem = experiment_basic_inference()
    
    experiment_quantization_comparison(base_mem, base_load_time, base_inf_time)
    
    print("\n所有实验完成!")
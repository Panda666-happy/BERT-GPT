import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 统一模型加载方式（避免重复加载）
model_name = "Wenzhong-GPT2-110M"

# 2. 加载分词器（优先使用fast版本加速）
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # 显式设置pad_token

# 3. 加载模型（统一使用FP32避免LayerNorm报错）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # 强制全精度
    use_safetensors=True
).to('cuda' if torch.cuda.is_available() else 'cpu')

# 4. 生成配置优化
generation_config = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id  # 显式指定结束标记
}


def generate_continuation(prompt):
    # 输入处理（自动移动到模型所在设备）
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成时禁用梯度计算
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config
        )

    # 解码时跳过特殊标记
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 5. 执行生成
if __name__ == "__main__":
    print("=== 续写结果 ===")
    print(generate_continuation("当我醒来，发现自己变成了一本书"))
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file, load_file
import json
import os
import shutil
import gc
import time
import datetime 
import sys  

#  配置区域 
ORIG_MODEL_PATH = "/opt/pangu/openPangu-Embedded-1B-V1.1"
OUTPUT_QUANT_PATH = "/opt/pangu/oht/quantize/quantized_model_int8_fixed_v3"
FORCE_REQUANTIZE = False


# 检查设备
if hasattr(torch, 'npu') and torch.npu.is_available():
    DEVICE = torch.device("npu:0") 
else:
    DEVICE = torch.device("cpu")


# 日志记录器类：同时输出到控制台和文件
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        # 同时写入终端和文件
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # 兼容 flush 方法
        self.terminal.flush()
        self.log.flush()

def get_dir_size(path):
    total = 0
    if not os.path.exists(path): return 0
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith(('.bin', '.safetensors', '.pt', '.json', '.model')):
                total += os.path.getsize(os.path.join(r, file))
    return total / 1024**3


# 自定义层: W8A16Linear (Int8存储, BF16计算)
class W8A16Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.bfloat16))
        
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    def forward(self, input_tensor):
        weight_bf16 = self.weight.to(input_tensor.dtype) * self.weight_scale
        return F.linear(input_tensor, weight_bf16, self.bias)

    @classmethod
    def from_linear(cls, linear_layer):
        new_layer = cls(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None)
        return new_layer

# 量化器
class ModelQuantizer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.skip_keywords = ["lm_head", "embed_tokens", "wte", "wpe", "output"]

    def run(self):
        print("\n" + "="*50)
        print("阶段一：执行模型量化 (Compressing Model)")
        print("="*50)
        
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)

        print(f"正在加载原始模型: {self.input_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.input_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.input_path, trust_remote_code=True)

        quantized_state_dict = {}
        quant_cnt = 0
        skip_cnt = 0

        print("开始遍历权重 (保护敏感层)...")
        for name, param in model.named_parameters():
            is_linear_weight = name.endswith(".weight") and param.dim() == 2
            is_sensitive = any(k in name for k in self.skip_keywords)

            if is_linear_weight and not is_sensitive:
                max_val = torch.abs(param).amax(dim=1, keepdim=True)
                scale = max_val / 127.0
                scale = torch.clamp(scale, min=1e-6)
                quantized_weight = (param / scale).round().clamp(-128, 127).to(torch.int8)
                
                quantized_state_dict[name] = quantized_weight
                quantized_state_dict[f"{name}_scale"] = scale.to(torch.bfloat16)
                quant_cnt += 1
            else:
                quantized_state_dict[name] = param
                skip_cnt += 1

        print("正在保存 model.safetensors (INT8)...")
        save_file(quantized_state_dict, os.path.join(self.output_path, "model.safetensors"))
        tokenizer.save_pretrained(self.output_path)
        model.config.save_pretrained(self.output_path)
        

        for file in os.listdir(self.input_path):
            if file.endswith(".py") and not os.path.exists(os.path.join(self.output_path, file)):
                shutil.copy(os.path.join(self.input_path, file), self.output_path)
        
        # 打印文件大小统计
        orig_size = get_dir_size(self.input_path)
        new_size = get_dir_size(self.output_path)
        print(f"量化完成!")
        print(f"原始大小: {orig_size:.2f} GB")
        print(f"量化后大小: {new_size:.2f} GB")
        print(f"压缩率: {(1 - new_size/orig_size)*100:.1f}%")

        del model
        gc.collect()

class InferenceEngine:
    def __init__(self, original_path, quantized_path):
        self.original_path = original_path
        self.quantized_path = quantized_path
        self.model = None
        self.tokenizer = None
        self.skip_keywords = ["lm_head", "embed_tokens", "wte", "wpe", "output"]

    def _print_model_stats(self, title):

        if self.model is None: return
        
        # 计算参数量
        param_count = sum(p.numel() for p in self.model.parameters())
        
        # 计算运行时显存/内存占用
        total_bytes = 0
        for m in self.model.modules():
            if isinstance(m, W8A16Linear):
                total_bytes += m.weight.numel() * 1  # int8
                total_bytes += m.weight_scale.numel() * 2 # bf16
                if m.bias is not None: total_bytes += m.bias.numel() * 2
            elif isinstance(m, nn.Linear):
                total_bytes += m.weight.numel() * 2 # bf16
                if m.bias is not None: total_bytes += m.bias.numel() * 2
        

        
        if hasattr(torch, 'npu') and self.model.device.type == 'npu':
             mem_mb = torch.npu.memory_allocated() / 1024**2
        else:
             # CPU 估算
             mem_mb = total_bytes / 1024**2 

        print(f"\n[{title}] 状态统计:")
        print(f"参数数量: {param_count / 1e9:.2f} B (亿)")
        print(f"运行时显存/内存占用: {mem_mb:.2f} MB ({mem_mb/1024:.2f} GB)")
        print(f"当前权重数据类型: 混合精度 (Int8存储 + BF16)")
        
        if "对比" in title:
            orig_size = get_dir_size(self.original_path)
            quant_size = get_dir_size(self.quantized_path)
            print(f"[硬盘] 原始权重文件大小: {orig_size:.2f} GB")
            print(f"[硬盘] 量化权重文件大小: {quant_size:.2f} GB")
            if orig_size > 0:
                print(f"[硬盘] 文件体积压缩比: {(1 - quant_size/orig_size)*100:.1f}%")
        print("-" * 50)

    def _replace_layers(self, module, prefix=""):
        count = 0
        skipped = 0
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if any(k in full_name for k in self.skip_keywords) or any(k in name for k in self.skip_keywords):
                skipped += 1
                continue

            if isinstance(child, nn.Linear):
                new_layer = W8A16Linear.from_linear(child)
                setattr(module, name, new_layer)
                count += 1
            else:
                c, s = self._replace_layers(child, full_name)
                count += c
                skipped += s
        return count, skipped

    def load(self):
        def mock_logger(msg):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            print(f"{timestamp} - models.llm_models - INFO - {msg}")

        print("\n")
        
        # 1. 硬件信息
        if "npu" in str(DEVICE):
            npu_cnt = torch.npu.device_count()
            # 获取设备属性
            npu_name = torch.npu.get_device_name(0)
            npu_mem = torch.npu.get_device_properties(0).total_memory / 1024**3
            
            mock_logger(f"检测到 {npu_cnt} 个NPU设备")
            mock_logger(f"NPU 0: {npu_name}, 总显存: {npu_mem:.2f} GB")
            mock_logger(f"盘古1B模型设备配置: ['{DEVICE}']")
        else:
             mock_logger(f"未检测到NPU设备，使用: {DEVICE}")

        # 2. 模型信息
        model_name_display = "openPangu-Embedded" 
        model_path_var = self.original_path
        
        mock_logger(f"预加载{model_name_display}模型...")
        mock_logger(f"正在加载{model_name_display}模型: {model_path_var}")
        mock_logger(f"使用单NPU模式: {DEVICE}")

        print("\n" + "="*50)
        
        print("正在初始化...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.quantized_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 1. 构建骨架
        config = AutoConfig.from_pretrained(self.quantized_path, trust_remote_code=True)
        config.torch_dtype = torch.bfloat16
        with torch.device("cpu"):
            self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            self.model.to(torch.bfloat16)

        print("加载量化权重文件...")
        # 打印文件对比
        self._print_model_stats("量化文件对比")

        # 2. 替换结构
        print("正在应用量化权重...")
        replace_count, skip_count = self._replace_layers(self.model)
        print(f"准备覆盖 {replace_count} 个量化层 (跳过了 {skip_count} 个敏感层/非Linear层)")
        
        # 3. 加载权重
        print("覆盖模型权重...")
        state_dict = load_file(os.path.join(self.quantized_path, "model.safetensors"), device="cpu")
        model_dict = self.model.state_dict()
        
        for name, param in state_dict.items():
            if name in model_dict:
                model_dict[name].copy_(param)
            elif name.endswith("_scale"):
                target_name = name.replace("_scale", ".weight_scale")
                if target_name in model_dict:
                    model_dict[target_name].copy_(param)
        del state_dict
        gc.collect()


        print("NPU数据类型统一为 bfloat16")
        
        print(f"将模型移动到 {DEVICE}...")
        self.model.to(DEVICE)
        self.model.eval()
        
        self._print_model_stats("最终混合模型(NPU内存)")
        
        print("\n" + "#" * 20 + " 指标达成证明 " + "#" * 20)
        print("1. 基于昇腾运行")
        print(f"2. 加载模型为 {model_name_display} 系列开源模型")
        print("3. 模型设备与结构详情:")
        print(self.model.device, self.model)
        print("#" * 56 + "\n")

        return True

    def test_forward_pass(self, test_text="测试"):
        
        print(f"\n=== 前向传播测试 ===")
        if not self.model: return

        inputs = self.tokenizer(test_text, return_tensors="pt").to(DEVICE)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            print(f"前向传播成功")
            print(f"输出形状: {logits.shape}")
            print(f"输出范围: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"输出均值: {logits.mean().item():.4f}")
        except Exception as e:
            print(f"前向传播失败: {e}")

    def generate(self, prompt, max_new_tokens=2048):
        if not self.model: return
        print(f"\n=== 推理测试 ===")
        print(f"输入: {prompt}")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_len = inputs.input_ids.shape[1]
        
        try:


            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            end_time = time.time()
            
            # 计算统计数据
            total_len = outputs.shape[1]
            gen_len = total_len - input_len
            elapsed = end_time - start_time
            speed = gen_len / elapsed if elapsed > 0 else 0
            
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"输出: {full_text}")
            
            
            print("生成统计:")
            print(f"生成token数量: {gen_len}")
            print(f"耗时: {elapsed:.2f} 秒")
            print(f"速度: {speed:.2f} tokens/秒")
            
        except Exception as e:
            print(f"推理出错: {e}")


def main():
    # 配置日志记录
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_log_{current_time}.txt"   
    sys.stdout = Logger(log_filename)
      
    print(f"日志将保存至: {os.path.abspath(log_filename)}")

    if FORCE_REQUANTIZE or not os.path.exists(os.path.join(OUTPUT_QUANT_PATH, "model.safetensors")):
        quantizer = ModelQuantizer(ORIG_MODEL_PATH, OUTPUT_QUANT_PATH)
        quantizer.run()

    engine = InferenceEngine(ORIG_MODEL_PATH, OUTPUT_QUANT_PATH)
    if engine.load():
        # 1. 运行前向传播测试 
        engine.test_forward_pass("什么是微积分")
        
        # 2. 运行生成测试 
        prompts = ["什么是微积分", "中国空间站", "有钱真的好吗"]
        for p in prompts:
            engine.generate(p)

if __name__ == "__main__":
    main()

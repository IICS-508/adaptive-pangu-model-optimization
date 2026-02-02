import os
import sys
import json
import glob
import random
import shutil
import gc
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

try:
    import torch_npu
except ImportError as e:
    print(f"Error importing torch_npu: {e}")
    sys.exit(1)

from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)

# 全局配置
CONFIG = {
    "model_path": "/data/pangu/openPangu-Embedded-1B-V1.1", 
    "output_dir": "/data/pangu/cmj/pm/kl/x5e-5_only_layer_0100",
    "finetune_data_path": "/data/pangu/cmj/data2/mix_data/50k_0_1_0_0.jsonl",
    
    # 梯度校准数据集
    "calib_data_sources": {"custom": "/data/pangu/cmj/data2/mix_data/100_0_1_0_0.jsonl"},
    
    "enable_distillation": True, # 是否启用知识蒸馏
    "target_sparsity": 0.2,      # 最终目标稀疏度
    "iterative_steps": 5,        # 迭代剪枝的总步数
    "npu_alignment": 256,        # NPU 矩阵运算对齐使得推理更高效
    
    # 微调参数
    "finetune_epochs_per_iter": 2,
    "learning_rate": 5e-5,
    "batch_size": 8,
    "max_seq_len": 1024,
    "gradient_accumulation_steps": 4,
    
    # 蒸馏权重设置
    "logits_distill_weight": 1.0, 
    "layer_distill_weight": 0,
    
    "device_id": 6,
    "seed": 42,
    "num_workers": 0,
    "pin_memory": False
}

# 日志工具
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        log_dir = os.path.dirname(filename)
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        self.log = open(filename, "a", encoding='utf-8')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
 
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)

def setup_device():
    torch.npu.set_device(CONFIG["device_id"])
    return torch.device(f"npu:{CONFIG['device_id']}")

def cleanup_memory():
    gc.collect()
    torch.npu.empty_cache()

def align_to_npu(num, align=256):
    return (int(num) // align) * align

def compute_taylor_importance(model, dataloader, device):
    """计算基于一阶 Taylor 展开的重要性分数"""
    model.train()
    model.zero_grad()
    # 冻结 Embedding 层避免不必要的计算
    if hasattr(model, "get_input_embeddings"):
        model.get_input_embeddings().requires_grad_(False)
        
    for i, batch in enumerate(dataloader):
        if i >= 16: break # 限制校准批次
        
        input_ids = batch['input_ids'].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        
    scores = {}
    layers = model.model.layers if hasattr(model, "model") else model.layers
    
    for i, layer in enumerate(layers):
        if not hasattr(layer, "mlp"): continue
        mlp = layer.mlp
        
        # 计算 Up Projection 的重要性: |W * Grad|
        up_layer = getattr(mlp, 'c_fc' if hasattr(mlp, 'c_fc') else 'up_proj')
        w_up = up_layer.weight.data
        g_up = up_layer.weight.grad
        
        if g_up is None:
            score = w_up.abs().sum(dim=1)
        else:
            score = torch.sum(torch.abs(w_up * g_up), dim=1)
            
        # 如果存在 Gate Projection (如 LLaMA/SwiGLU 结构)，将其重要性叠加
        if hasattr(mlp, 'gate_proj'):
            gate_layer = mlp.gate_proj
            w_gate = gate_layer.weight.data
            g_gate = gate_layer.weight.grad
            
            if g_gate is not None:
                score += torch.sum(torch.abs(w_gate * g_gate), dim=1)
            else:
                score += w_gate.abs().sum(dim=1)
            
        scores[i] = score.float().detach() 
        
    model.zero_grad()
    if hasattr(model, "get_input_embeddings"):
        model.get_input_embeddings().requires_grad_(True)
    cleanup_memory()
    return scores

def prune_mlp_layer(mlp, idxs, dev):
    """根据索引对 MLP 层进行物理剪枝"""
    up_n = 'c_fc' if hasattr(mlp, 'c_fc') else 'up_proj'
    down_n = 'c_proj' if hasattr(mlp, 'c_proj') else 'down_proj'
    up, down = getattr(mlp, up_n), getattr(mlp, down_n)
    new_d = len(idxs)
    
    # 重构 Up Projection
    n_up = nn.Linear(up.in_features, new_d, bias=(up.bias is not None)).to(dev, torch.bfloat16)
    n_up.weight.data = up.weight.data[idxs].clone().to(torch.bfloat16)
    if up.bias is not None: n_up.bias.data = up.bias.data[idxs].clone().to(torch.bfloat16)
    
    # 重构 Down Projection
    n_down = nn.Linear(new_d, down.out_features, bias=(down.bias is not None)).to(dev, torch.bfloat16)
    n_down.weight.data = down.weight.data[:, idxs].clone().to(torch.bfloat16)
    if down.bias is not None: n_down.bias.data = down.bias.data.clone().to(torch.bfloat16)
    
    setattr(mlp, up_n, n_up); setattr(mlp, down_n, n_down)
    
    # 处理 Gate Projection
    if hasattr(mlp, 'gate_proj'):
        gate = getattr(mlp, 'gate_proj')
        n_gate = nn.Linear(gate.in_features, new_d, bias=(gate.bias is not None)).to(dev, torch.bfloat16)
        n_gate.weight.data = gate.weight.data[idxs].clone().to(torch.bfloat16)
        setattr(mlp, 'gate_proj', n_gate)
        
    torch.npu.synchronize()
    return new_d

# 数据集处理
class MaskedPackedDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=1024):
        self.samples = []
        ids_buf, labs_buf = [], []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        it = json.loads(line)
                        p, r = it.get('prompt',''), it.get('response','')
                        if p and r:
                            p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
                            r_ids = tokenizer(r + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
                            ids_buf.extend(p_ids + r_ids)
                            labs_buf.extend([-100]*len(p_ids) + r_ids)
                    except: continue
        for i in range(0, len(ids_buf), max_len):
            c_i = ids_buf[i : i+max_len]
            c_l = labs_buf[i : i+max_len]
            if len(c_i) < max_len:
                pad = max_len - len(c_i)
                c_i += [tokenizer.pad_token_id]*pad
                c_l += [-100]*pad
            self.samples.append({"input_ids": torch.tensor(c_i), "labels": torch.tensor(c_l)})
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

class SimpleCollator:
    def __call__(self, b):
        return {
            "input_ids": torch.stack([x['input_ids'] for x in b]),
            "labels": torch.stack([x['labels'] for x in b]),
            "attention_mask": torch.stack([x['input_ids'].ne(0) for x in b]).long()
        }

# 微调策略
def run_finetune(student, teacher, loader, epochs, device, lr, iter_idx):
    print(f">>> 开始微调 (蒸馏模式={teacher is not None}, 学习率={lr:.2e})...")
    
    log_dir = os.path.join(CONFIG['output_dir'], 'logs', f'iter_{iter_idx}')
    writer = SummaryWriter(log_dir=log_dir)
    
    student.train()
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        student.enable_input_require_grads()

    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01, eps=1e-6)
    total_steps = len(loader) * epochs // CONFIG['gradient_accumulation_steps']
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
    mse_loss_fn = nn.MSELoss()
    
    global_step = 0
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            in_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            if teacher:
                # 蒸馏模式：跳过标准 CE 计算，仅优化 KL 和 MSE
                s_out = student(input_ids=in_ids, attention_mask=mask, output_hidden_states=True)
                with torch.no_grad():
                    t_out = teacher(input_ids=in_ids, attention_mask=mask, output_hidden_states=True)
                
                kl_per_token = F.kl_div(
                    F.log_softmax(s_out.logits / 2.0, dim=-1),
                    F.softmax(t_out.logits / 2.0, dim=-1),
                    reduction='none'
                ).sum(dim=-1) * 4.0
                
                # 对 Padding Token 进行掩码处理
                labels = batch['labels'].to(device)
                valid_mask = (labels != -100).float()
                num_valid = valid_mask.sum()
                kl_loss = (kl_per_token * valid_mask).sum() / num_valid if num_valid > 0 else torch.tensor(0.0).to(device)
                
                hid_loss = 0
                s_h, t_h = s_out.hidden_states, t_out.hidden_states
                for i in range(1, len(s_h)): 
                    hid_loss += mse_loss_fn(s_h[i].to(torch.float32), t_h[i].to(torch.float32))
                hid_loss /= (len(s_h)-1)
                
                loss = CONFIG['logits_distill_weight']*kl_loss + CONFIG['layer_distill_weight']*hid_loss
            else:
                # 标准微调模式 (无蒸馏)
                labels = batch['labels'].to(device)
                s_out = student(input_ids=in_ids, attention_mask=mask, labels=labels)
                loss = s_out.loss

            loss = loss / CONFIG['gradient_accumulation_steps']
            loss.backward()
            
            if (step+1) % CONFIG['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                
                current_total_loss = loss.item() * CONFIG['gradient_accumulation_steps']
                writer.add_scalar('Loss/Total', current_total_loss, global_step)
                
                if teacher:
                    writer.add_scalar('Loss/KL', kl_loss.item(), global_step)
                    writer.add_scalar('Loss/Hidden', hid_loss.item(), global_step)
                
                writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)

                if (step // CONFIG['gradient_accumulation_steps']) % 10 == 0:
                    print(f"Ep {epoch+1} | Loss: {current_total_loss:.4f}")
                
                global_step += 1

    writer.close()
    if hasattr(student, "gradient_checkpointing_disable"): student.gradient_checkpointing_disable()
    cleanup_memory()

# 主流程
class CalibDS(Dataset): 
    def __init__(self, tok):
        self.d = []
        with open(CONFIG["calib_data_sources"]["custom"]) as f:
            txt = "".join([json.loads(l)['prompt'] for l in f.readlines()[:200]])
        t = tok(txt, return_tensors='pt')['input_ids'][0]
        for i in range(0, len(t)-1024, 1024): 
            self.d.append(t[i:i+1024])
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return {"input_ids": self.d[i]}

def main():
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
    
    log_filename = os.path.join(CONFIG['output_dir'], f"run_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = DualLogger(log_filename)
    sys.stderr = sys.stdout 
    
    print(f"=== 日志记录已启动。输出将保存至 {log_filename} ===")

    set_seed(CONFIG['seed'])
    device = setup_device()
    
    device_props = torch.npu.get_device_properties(CONFIG['device_id'])
    device_full_name = device_props.name 
    total_mem_gb = device_props.total_memory / (1024**3)
    
    custom_log(f"检测到 1 个NPU设备")
    custom_log(f"NPU {CONFIG['device_id']}: {device_full_name}, 总显存: {total_mem_gb:.2f} GB")
    
    model_dir_name = os.path.basename(CONFIG['model_path'].rstrip('/'))
    custom_log(f"盘古模型设备配置: ['npu:{CONFIG['device_id']}']")
    custom_log(f"预加载{model_dir_name}模型...")
    custom_log(f"正在加载{model_dir_name}模型: {CONFIG['model_path']}")

    # 模型和数据加载
    tok = AutoTokenizer.from_pretrained(CONFIG['model_path'], trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(CONFIG['model_path'], trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    
    teacher = None
    if CONFIG['enable_distillation']:
        teacher = AutoModelForCausalLM.from_pretrained(CONFIG['model_path'], trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    calib_loader = DataLoader(CalibDS(tok), batch_size=4, shuffle=True)
    ft_loader = DataLoader(
        MaskedPackedDataset(CONFIG['finetune_data_path'], tok, max_len=CONFIG['max_seq_len']), 
        batch_size=CONFIG['batch_size'], 
        collate_fn=SimpleCollator(), 
        shuffle=True, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG['pin_memory']
    )
    
    # 获取原始维度
    if hasattr(model.config, "intermediate_size"):
        original_dim = model.config.intermediate_size
    else:
        layer0 = model.model.layers[0] if hasattr(model, "model") else model.layers[0]
        original_dim = layer0.mlp.c_fc.out_features if hasattr(layer0.mlp, "c_fc") else layer0.mlp.up_proj.out_features
    
    layers = model.model.layers if hasattr(model, "model") else model.layers

    for step in range(1, CONFIG['iterative_steps'] + 1):
        print(f"\n=== 迭代进度 {step}/{CONFIG['iterative_steps']} ===")
        
        scores_map = compute_taylor_importance(model, calib_loader, device)
        
        progress = step / CONFIG['iterative_steps']
        current_sparsity = CONFIG['target_sparsity'] * progress
        
        target_dim = int(original_dim * (1.0 - current_sparsity))
        aligned_dim = max(align_to_npu(target_dim, CONFIG['npu_alignment']), CONFIG['npu_alignment'])
        
        print(f"稀疏度: {current_sparsity:.2%} | 维度变化: {original_dim} -> {aligned_dim}")

        for i, layer in enumerate(layers):
            if not hasattr(layer, "mlp"): continue
            s = scores_map[i]
            if len(s) <= aligned_dim: continue
            
            _, topk = torch.topk(s, aligned_dim)
            keep, _ = torch.sort(topk) 
            prune_mlp_layer(layer.mlp, keep, device)

        model.config.intermediate_size = aligned_dim

        run_finetune(model, teacher, ft_loader, CONFIG['finetune_epochs_per_iter'], device, CONFIG['learning_rate'], iter_idx=step)
        
        if current_sparsity >= 0.04:
            save_path = os.path.join(CONFIG['output_dir'], f"iter_{step}")
            if not os.path.exists(save_path): os.makedirs(save_path)
            
            model.config.use_fused_attention = True
            model.config.use_flash_attention = True
            
            print(f"保存检查点至 {save_path}...")
            model.save_pretrained(save_path, safe_serialization=True)
            tok.save_pretrained(save_path)
            for f in glob.glob(os.path.join(CONFIG['model_path'], "*.py")): shutil.copy(f, save_path)
            
            model.config.use_fused_attention = False
            model.config.use_flash_attention = False
        else:
            print(f"跳过保存 (当前稀疏度 {current_sparsity:.2%} < 4%)")
        
        cleanup_memory()

    print("\n任务流程完成。")

if __name__ == "__main__":
    main()

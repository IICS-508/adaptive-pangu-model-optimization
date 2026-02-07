import os
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from threading import Thread


if hasattr(torch, 'npu') and torch.npu.is_available():
    import torch_npu
    DEVICE = torch.device("npu:0")
    print(f"[Init] 检测到 NPU 设备: {torch.npu.get_device_name(0)}")


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextIteratorStreamer
from safetensors.torch import load_file


# 1B 小模型路径
PATH_ROUTER_1B = "/opt/pangu/openPangu-Embedded-1B-V1.1"

# 7B 基础模型路径
PATH_BASE_7B = "/opt/pangu/openPangu-Embedded-7B-V1.1"         

# Int8 量化模型路径
PATH_QUANT_INT8 = "/opt/pangu/oht/quantize/quantized_model_int8_fixed_v3"

# 剪枝模型路径
PATH_PRUNED = "/opt/pangu/pruned_model/openPangu-Pruned-7B"    

# 历史记录最大轮数
MAX_HISTORY_LEN = 3


def clean_memory():

    gc.collect()
    if "npu" in str(DEVICE):
        torch.npu.empty_cache()
        torch.npu.synchronize()

def print_npu_status(tag=""):
    if "npu" in str(DEVICE):
        mem = torch.npu.memory_allocated() / 1024**3
        print(f"   [NPU状态] {tag} | 当前显存占用: {mem:.2f} GB")


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

def replace_layers_with_quant(module, skip_keywords=["lm_head", "embed_tokens", "wte", "wpe", "output"]):
    for name, child in module.named_children():
        if any(k in name for k in skip_keywords): continue
        if isinstance(child, nn.Linear):
            setattr(module, name, W8A16Linear.from_linear(child))
        else:
            replace_layers_with_quant(child, skip_keywords)

def load_quantized_engine(path):
    print(f"   正在加载量化模型: {path}")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config.torch_dtype = torch.bfloat16
    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.to(torch.bfloat16)
    
    replace_layers_with_quant(model)
    
    state_dict = load_file(os.path.join(path, "model.safetensors"), device="cpu")
    model_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_dict:
            model_dict[name].copy_(param)
        elif name.endswith("_scale"):
            target_name = name.replace("_scale", ".weight_scale")
            if target_name in model_dict:
                model_dict[target_name].copy_(param)
    del state_dict
    model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return model, tokenizer


class NGramMatcher:
    def __init__(self, ngram_size=2, max_draft_len=4):
        self.ngram_size = ngram_size
        self.max_draft_len = max_draft_len

    def find_candidate(self, input_ids_list: list) -> list:
        if len(input_ids_list) < self.ngram_size: return []
        trigger = input_ids_list[-self.ngram_size:]
        search_limit = len(input_ids_list) - self.ngram_size - 1
        for i in range(search_limit, -1, -1):
            if input_ids_list[i : i + self.ngram_size] == trigger:
                start = i + self.ngram_size
                end = min(start + self.max_draft_len, len(input_ids_list) - self.ngram_size)
                res = input_ids_list[start : end]
                if res: return res
        return []


class AscendInteractiveChat:
    def __init__(self):
        self.router_model = None
        self.router_tokenizer = None
        self.worker_model = None
        self.worker_tokenizer = None
        
        self.current_worker_type = None  # 'quant', 'pld', 'pruned'
        self.history = []                # [(user, bot), ...]

    def init_system(self):

        print("\n" + "="*50)
        print("   正在启动 Ascend 智能对话系统...")
        clean_memory()
        try:
            print(f"   [System] 加载 Router (1B) ...")
            self.router_tokenizer = AutoTokenizer.from_pretrained(PATH_ROUTER_1B, trust_remote_code=True)
            self.router_model = AutoModelForCausalLM.from_pretrained(
                PATH_ROUTER_1B, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(DEVICE).eval()
            print("   [System] Router 加载成功。")
        except Exception as e:
            print(f"   [Error] Router 加载失败，请检查路径: {e}")
            exit(1)

    def analyze_intent(self, query):

        clean_memory()
        prompt = (
            "任务：分析用户输入的意图类别。\n"
            "输入：写个冒泡排序\n类别：代码编程\n"
            "输入：今天天气不错\n类别：日常闲聊\n"
            "输入：什么是相对论\n类别：知识问答\n"
            f"输入：{query}\n类别："
        )
        inputs = self.router_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.router_model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        response = self.router_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取最后生成的类别
        if "类别：" in response:
            category = response.split("类别：")[-1].strip()
        else:
            category = response[-10:].strip()
        return category

    def ask_user_strategy(self, category):

        print(f"\n{'-'*45}")
        print(f"  [Router分析] 意图识别为: 【{category}】")
        
        # 推荐逻辑
        if any(x in category for x in ["代码", "编程", "数学", "计算", "逻辑"]):
            recommend = "2"
            rec_msg = "(Router推荐: 逻辑强)"
        else:
            recommend = "1"
            rec_msg = "(Router推荐: 省显存)"

        print(f"  请选择推理模型:")
        print(f"  1. [量化] Int8 Quant   - {rec_msg if recommend=='1' else ''}")
        print(f"  2. [投机] PLD 7B       - {rec_msg if recommend=='2' else ' (速度快)'}")
        print(f"  3. [剪枝] Pruned 7B    - (轻量化)")
        print(f"  (按 Enter 使用推荐值, 或输入 1/2/3)")
        
        choice = input("  您的选择 > ").strip()
        print(f"{'-'*45}")

        if not choice: choice = recommend
        
        mapping = {"1": "quant", "2": "pld", "3": "pruned"}
        return mapping.get(choice, "quant")

    def switch_worker(self, target_type):

        if self.current_worker_type == target_type:
            return # 无需切换

        print(f"[System] 正在切换模型: {self.current_worker_type} -> {target_type}")
        
        # 1. 卸载旧模型
        if self.worker_model:
            del self.worker_model
            del self.worker_tokenizer
            self.worker_model = None
            clean_memory()
            print("   旧模型已卸载")

        # 2. 加载新模型
        start_t = time.time()
        try:
            if target_type == 'pld':
                print("   加载 Base 7B (for PLD)...")
                self.worker_tokenizer = AutoTokenizer.from_pretrained(PATH_BASE_7B, trust_remote_code=True)
                self.worker_model = AutoModelForCausalLM.from_pretrained(
                    PATH_BASE_7B, torch_dtype=torch.bfloat16, trust_remote_code=True
                ).to(DEVICE).eval()
            
            elif target_type == 'pruned':
                print("   加载 Pruned 7B...")
                if not os.path.exists(PATH_PRUNED): raise FileNotFoundError("剪枝模型路径不存在")
                self.worker_tokenizer = AutoTokenizer.from_pretrained(PATH_PRUNED, trust_remote_code=True)
                self.worker_model = AutoModelForCausalLM.from_pretrained(
                    PATH_PRUNED, torch_dtype=torch.bfloat16, trust_remote_code=True
                ).to(DEVICE).eval()

            else: # quant
                self.worker_model, self.worker_tokenizer = load_quantized_engine(PATH_QUANT_INT8)
            
            self.current_worker_type = target_type
            print(f"   模型加载完成，耗时 {time.time()-start_t:.2f}s")
            print_npu_status()
            
        except Exception as e:
            print(f"   [Error] 加载失败 ({e})，尝试回退到量化模型...")
            self.current_worker_type = 'quant'
            self.worker_model, self.worker_tokenizer = load_quantized_engine(PATH_QUANT_INT8)

    def build_prompt(self, query):

        context = ""
        # 仅取最近几轮，防止 Prompt 过长
        history_segment = self.history[-MAX_HISTORY_LEN:]
        
        for i, (u, b) in enumerate(history_segment):
            # [Round i] 这种强提示符有助于 Base 模型区分轮次
            context += f"[Round {i}]\nUser: {u}\nAssistant: {b}\n"
        
        current_idx = len(history_segment)
        final_prompt = f"{context}[Round {current_idx}]\nUser: {query}\nAssistant:"
        

        return final_prompt

    def generate_response(self, query):
        prompt = self.build_prompt(query)
        
        # 每次生成前，清理一下
        clean_memory()
        
        if self.current_worker_type == 'pld':
            return self._pld_stream(prompt)
        else:
            return self._normal_stream(prompt)


    def _normal_stream(self, prompt):
        inputs = self.worker_tokenizer(prompt, return_tensors="pt")
        # 简单的长度截断保护
        if inputs.input_ids.shape[1] > 2048:
            inputs = self.worker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        inputs = inputs.to(DEVICE)
        
        streamer = TextIteratorStreamer(self.worker_tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            streamer=streamer
        )
        
        Thread(target=self.worker_model.generate, kwargs=gen_kwargs).start()
        
        full_resp = ""
        print("Assistant: ", end="", flush=True)
        
        for text in streamer:
            # [关键修复] 停止词检查：如果模型自己生成了 User:，立即停止
            if "User:" in text or "[Round" in text:
                break
            
            print(text, end="", flush=True)
            full_resp += text
        print()
        return full_resp


    def _pld_stream(self, prompt):
        matcher = NGramMatcher(6, 2)
        inputs = self.worker_tokenizer(prompt, return_tensors="pt")
        if inputs.input_ids.shape[1] > 2048:
            inputs = self.worker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        input_ids = inputs.input_ids.to(DEVICE)
        input_ids_cpu = input_ids[0].cpu().tolist()
        eos_id = self.worker_tokenizer.eos_token_id
        
        print("Assistant (PLD): ", end="", flush=True)
        
        full_resp = ""
        generated = 0
        max_tokens = 512
        
        # 1. Prefill
        with torch.no_grad():
            out = self.worker_model(input_ids=input_ids, use_cache=True)
        kv, logits = out.past_key_values, out.logits[:, -1, :]
        
        while generated < max_tokens:
            drafts = matcher.find_candidate(input_ids_cpu)
            commit = []
            recomp = False

            if not drafts:
                # 常规生成
                token = torch.argmax(logits).item()
                commit.append(token)
                if generated < max_tokens:
                    with torch.no_grad():
                        out = self.worker_model(input_ids=torch.tensor([[token]], device=DEVICE), past_key_values=kv, use_cache=True)
                    kv, logits = out.past_key_values, out.logits[:, -1, :]
            else:
                # 投机验证
                model_token = torch.argmax(logits).item()
                if model_token != drafts[0]:
                    commit.append(model_token)
                    recomp = True
                else:
                    draft_t = torch.tensor([drafts], device=DEVICE)
                    with torch.no_grad():
                        out = self.worker_model(input_ids=draft_t, past_key_values=kv, use_cache=True)
                    
                    commit.append(drafts[0])
                    d_logits = out.logits
                    stop = (drafts[0] == eos_id)
                    all_match = True
                    
                    for i in range(len(drafts)-1):
                        if stop: break
                        pred = torch.argmax(d_logits[0, i, :]).item()
                        if pred == drafts[i+1]:
                            commit.append(pred)
                            if pred == eos_id: stop = True
                        else:
                            commit.append(pred)
                            all_match = False
                            recomp = True
                            break
                    
                    if all_match and not stop:
                        bonus = torch.argmax(d_logits[0, -1, :]).item()
                        commit.append(bonus)
                        if bonus != eos_id:
                            with torch.no_grad():
                                out_b = self.worker_model(input_ids=torch.tensor([[bonus]], device=DEVICE), past_key_values=out.past_key_values, use_cache=True)
                            kv, logits = out_b.past_key_values, out_b.logits[:, -1, :]
                            recomp = False
                        else: recomp = False
                    elif not all_match:
                        recomp = True

            if commit:
                txt_chunk = self.worker_tokenizer.decode(commit, skip_special_tokens=True)
                

                stop_flag = False
                valid_chunk = txt_chunk
                if "User:" in txt_chunk:
                    valid_chunk = txt_chunk.split("User:")[0]
                    stop_flag = True
                elif "[Round" in txt_chunk:
                    valid_chunk = txt_chunk.split("[Round")[0]
                    stop_flag = True
                
                print(valid_chunk, end="", flush=True)
                full_resp += valid_chunk
                
                if stop_flag: 
                    break

                generated += len(commit)
                input_ids_cpu.extend(commit)
                
                if recomp:
                    new_in = torch.tensor([commit], device=DEVICE)
                    with torch.no_grad():
                        out = self.worker_model(input_ids=new_in, past_key_values=kv, use_cache=True)
                    kv, logits = out.past_key_values, out.logits[:, -1, :]

                if eos_id in commit: break

        print()
        return full_resp

    def start(self):
        self.init_system()
        print("\n" + "="*50)
        print("系统就绪。")
        print("流程：输入问题 -> Router分析 -> 你选择模型 -> 生成回答")
        print("输入 'exit' 或 'q' 退出")
        print("="*50)

        while True:
            try:
                query = input("\n[User] 请输入: ").strip()
                if not query: continue
                if query.lower() in ["exit", "q", "quit"]: break
                
                # 1. 分析
                cat = self.analyze_intent(query)
                
                # 2. 决策
                strat = self.ask_user_strategy(cat)
                
                # 3. 切换
                self.switch_worker(strat)
                
                # 4. 生成
                resp = self.generate_response(query)
                
                # 5. 记录历史
                self.history.append((query, resp))
                
            except KeyboardInterrupt:
                print("\n[System] 检测到中断，停止当前生成。")
            except Exception as e:
                print(f"\n[System] 发生错误: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    chat_app = AscendInteractiveChat()
    chat_app.start()

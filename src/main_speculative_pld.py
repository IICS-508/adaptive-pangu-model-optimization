import os
import time
import datetime
import logging
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from opencompass.models.base import BaseModel
#eval.py调用本文件完成测评

logger = logging.getLogger("PLD_Experiment")
logger.setLevel(logging.INFO)

class NGramMatcher:

    def __init__(self, ngram_size=2, max_draft_len=4):
        self.ngram_size = ngram_size
        self.max_draft_len = max_draft_len

    def find_candidate(self, input_ids_list: list) -> list:
        if len(input_ids_list) < self.ngram_size:
            return []
        trigger = input_ids_list[-self.ngram_size:]
        seq_len = len(input_ids_list)
        search_limit = seq_len - self.ngram_size - 1
        for i in range(search_limit, -1, -1):
            if input_ids_list[i : i + self.ngram_size] == trigger:
                start_idx = i + self.ngram_size
                end_idx = min(start_idx + self.max_draft_len, seq_len - self.ngram_size) 
                candidate = input_ids_list[start_idx : end_idx]
                if candidate:
                    return candidate
        return []

class OpenCompassPLDModel(BaseModel):
    def __init__(self, 
                 path: str, 
                 tokenizer_path: str = None,
                 enable_pld: bool = False, 
                 ngram_size: int = 6, 
                 max_draft_len: int = 2, 
                 test_name: str = "General_Test",
                 model_kwargs: dict = None,
                 tokenizer_kwargs: dict = None,
                 generation_kwargs: dict = None, 
                 max_out_len: int = 512,
                 max_seq_len: int = 32768,       
                 device: str = "npu:0",
                 meta_template: dict = None,     
                 **kwargs):
        
        super().__init__(path=path, 
                         max_seq_len=max_seq_len, 
                         tokenizer_only=False, 
                         meta_template=meta_template)
        
        self.enable_pld = enable_pld
        self.ngram_size = ngram_size
        self.max_draft_len = max_draft_len
        self.max_out_len = max_out_len
        self.generation_kwargs = generation_kwargs or {}
        self.test_name = test_name
        
        self.device_str = device
        self.device = torch.device(device)
        self.path = path
        

        self._configure_logger()

        load_tk_path = tokenizer_path if tokenizer_path else path
        tk_args = tokenizer_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(load_tk_path, **tk_args)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading Model from {path}...")
        mod_args = model_kwargs or {}
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(path, **mod_args).to(self.device).eval()
        
        self.matcher = NGramMatcher(ngram_size, max_draft_len)


        self._log_system_info()

    def _configure_logger(self, work_dir='./'):

        mode_str = "PLD" if self.enable_pld else "Baseline"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        

        log_filename = os.path.join(work_dir, f"{mode_str}_{self.test_name}_{timestamp}.log")
        

        if logger.hasHandlers():
            logger.handlers.clear()
            
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        self.log_file_path = log_filename

    def _log_system_info(self):

        try:

            device_props = torch.npu.get_device_properties(self.device)
            device_name = torch.npu.get_device_name(self.device)
            

            total_mem_gb = device_props.total_memory / (1024 ** 3)
            allocated_gb = torch.npu.memory_allocated(self.device) / (1024 ** 3)
            reserved_gb = torch.npu.memory_reserved(self.device) / (1024 ** 3)
            
            mode_tag = "PLD Enabled" if self.enable_pld else "Baseline"
            
            info_msg = (
                f"\n{'='*25} ASCEND ENVIRONMENT INFO {'='*25}\n"
                f"Running Mode      : {mode_tag}\n"
                f"Test Name         : {self.test_name}\n"
                f"Model Path        : {self.path}\n"
                f"Device ID         : {self.device_str}\n"
                f"Device Name       : {device_name}\n"
                f"Total VRAM        : {total_mem_gb:.2f} GB\n"
                f"Current Allocated : {allocated_gb:.2f} GB\n"
                f"Current Reserved  : {reserved_gb:.2f} GB\n"
                f"Log Saved to      : {self.log_file_path}\n"
                f"{'='*75}"
            )
            logger.info(info_msg)
            print(info_msg)
        except Exception as e:

            logger.error(f"Failed to retrieve NPU system info: {e}")
            print(f"Warning: Could not retrieve NPU info ({e})")

    def _pld_generate(self, input_ids_tensor, max_new_tokens):

        input_ids = input_ids_tensor
        input_ids_cpu = input_ids[0].cpu().tolist()
        
        past_key_values = None
        generated_count = 0
        eos_token_id = self.tokenizer.eos_token_id
        
        # Prefill
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)
        
        prev_logits = outputs.logits[:, -1, :] 
        past_key_values = outputs.past_key_values
        
        while generated_count < max_new_tokens:
            draft_tokens_list = self.matcher.find_candidate(input_ids_cpu)
            tokens_to_commit = []
            need_recompute_context = False
            
            if not draft_tokens_list:
                next_token = torch.argmax(prev_logits, dim=-1, keepdim=True)
                token_val = next_token.item()
                tokens_to_commit.append(token_val)
                
                if generated_count + 1 < max_new_tokens and token_val != eos_token_id:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=next_token, 
                            past_key_values=past_key_values, 
                            use_cache=True
                        )
                    prev_logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                
            else:

                model_pred_1 = torch.argmax(prev_logits).item()
                draft_1 = draft_tokens_list[0]
                
                if model_pred_1 != draft_1:
                    tokens_to_commit.append(model_pred_1)
                    need_recompute_context = True
                else:
                    draft_tensor = torch.tensor([draft_tokens_list], device=self.device)
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=draft_tensor,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                    
                    tokens_to_commit.append(draft_1)
                    draft_logits = outputs.logits
                    all_match = True
                    stop_flag = False
                    
                    if draft_1 == eos_token_id:
                        stop_flag = True
                        all_match = False
                    
                    for i in range(len(draft_tokens_list) - 1):
                        if stop_flag: break
                        pred_next = torch.argmax(draft_logits[0, i, :]).item()
                        target_next = draft_tokens_list[i+1]
                        
                        if pred_next == target_next:
                            tokens_to_commit.append(target_next)
                            if target_next == eos_token_id:
                                stop_flag = True
                                all_match = False
                        else:
                            tokens_to_commit.append(pred_next)
                            all_match = False 
                            need_recompute_context = True 
                            break 
                    
                    if all_match and not stop_flag:
                        bonus_token = torch.argmax(draft_logits[0, -1, :]).item()
                        tokens_to_commit.append(bonus_token)
                        need_recompute_context = False 
                        
                        if bonus_token != eos_token_id:
                            with torch.no_grad():
                                bonus_out = self.model(
                                    input_ids=torch.tensor([[bonus_token]], device=self.device),
                                    past_key_values=outputs.past_key_values,
                                    use_cache=True
                                )
                            past_key_values = bonus_out.past_key_values
                            prev_logits = bonus_out.logits[:, -1, :]
                        
                    elif not all_match:
                        need_recompute_context = True

            if tokens_to_commit:
                final_tensor = torch.tensor([tokens_to_commit], device=self.device)
                input_ids = torch.cat([input_ids, final_tensor], dim=1)
                input_ids_cpu.extend(tokens_to_commit)
                generated_count += len(tokens_to_commit)
                
                if need_recompute_context:
                    with torch.no_grad():
                        res = self.model(
                            input_ids=final_tensor, 
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                    past_key_values = res.past_key_values
                    prev_logits = res.logits[:, -1, :]
                
                if eos_token_id in tokens_to_commit:
                    break

        return input_ids

    def _baseline_generate(self, input_ids_tensor, max_new_tokens):
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids_tensor, 
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **self.generation_kwargs 
            )
        return output_ids

    def generate(self, inputs: list[str], max_out_len: int = 512, **kwargs) -> list[str]:
        generated_texts = []
        mode_tag = "PLD" if self.enable_pld else "Baseline"
        
        for i, prompt in enumerate(inputs):
            input_tensor = self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids
            target_len = max_out_len if max_out_len else self.max_out_len
            

            torch.npu.synchronize()
            start_t = time.time()
            
            try:
                if self.enable_pld:
                    final_ids = self._pld_generate(input_tensor, max_new_tokens=target_len)
                else:
                    final_ids = self._baseline_generate(input_tensor, max_new_tokens=target_len)
            except RuntimeError as e:

                logger.error(f"RuntimeError during generation sample {i}: {e}")
                raise e
            
            
            torch.npu.synchronize()
            end_t = time.time()
            
            new_tokens_count = final_ids.shape[1] - input_tensor.shape[1]
            cost_time = end_t - start_t
            current_speed = new_tokens_count / cost_time if cost_time > 0 else 0
            
            log_msg = (
                f"[{mode_tag} | {self.test_name}] Sample {i+1} | "
                f"Tokens: {new_tokens_count} | "
                f"Time: {cost_time:.4f}s | "
                f"Speed: {current_speed:.2f} tokens/s"
            )
            logger.info(log_msg)
            print(log_msg)
            
            text = self.tokenizer.decode(final_ids[0], skip_special_tokens=True)
            
            if text.startswith(prompt):
                text = text[len(prompt):]
            elif len(text) > len(prompt) and text[:20] == prompt[:20]: 
                text = text[len(prompt):]

            generated_texts.append(text)
            
        return generated_texts

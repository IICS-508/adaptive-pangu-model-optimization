import os
import time
import torch
import glob
import pandas as pd
import logging
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Tuple, List, Dict

# 配置日志记录器
def setup_logger(output_dir, log_file_name="running.log"):
    logger = logging.getLogger("SpeculativeEval")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(output_dir, log_file_name), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

class TopKSpeculativeDecoder:
    def __init__(self, main_model, draft_model, tokenizer, device=None, k=5, max_draft_len=3, 
                 temperature=0.8, top_p=0.9, acceptance_threshold=0.01, logger=None):
        self.main_model = main_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.k = k
        self.max_draft_len = max_draft_len
        self.temperature = temperature
        self.top_p = top_p
        self.acceptance_threshold = acceptance_threshold
        self.logger = logger if logger else logging.getLogger("SpeculativeEval")
        

        if device:
            self.device = device
        else:
            self.device = torch.device("npu:5" if torch.npu.is_available() else "cpu")
        

        
        # 确保模型在正确的设备上
        if self.main_model.device != self.device:
            self.main_model = self.main_model.to(self.device)
        if self.draft_model.device != self.device:
            self.draft_model = self.draft_model.to(self.device)
        
        # 特殊token处理
        self.special_tokens_to_filter = set(tokenizer.all_special_ids)
        if tokenizer.eos_token_id in self.special_tokens_to_filter:
            self.special_tokens_to_filter.remove(tokenizer.eos_token_id)
        
        # 统计信息
        self.stats = {
            'draft_calls': 0,
            'main_calls': 0,
            'total_tokens': 0,
            'accepted_tokens': 0,
            'rejection_events': 0,
            'early_stop_count': 0,
            'generated_tokens': 0
        }

    def _ensure_npu_tensor(self, tensor):

        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor

    def _top_k_sampling(self, logits: torch.Tensor) -> torch.Tensor:

        top_k = min(self.k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        threshold = v[:, -1].unsqueeze(1)
        logits = torch.where(logits < threshold, torch.tensor(-float('inf'), device=self.device), logits)
        
        # 应用温度
        if self.temperature > 0:
            logits = logits / self.temperature
        
        # 应用top_p过滤
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # 删除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # 保留第一个超过top_p的token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 将被移除的token设为-inf
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float('inf')
        
        probs = torch.softmax(logits, dim=-1)
        return probs

    def _sample_token(self, probs: torch.Tensor) -> torch.Tensor:

        return torch.multinomial(probs, num_samples=1)

    def _check_early_stop(self, token: torch.Tensor) -> bool:

        return (token == self.tokenizer.eos_token_id).all().item()

    def _generate_draft_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        self.stats['draft_calls'] += 1
        input_ids = self._ensure_npu_tensor(input_ids)
        attention_mask = self._ensure_npu_tensor(attention_mask)
        
        batch_size = input_ids.shape[0]
        current_input = input_ids
        current_mask = attention_mask
        draft_tokens = []
        draft_probs_list = []
        
        # 计算当前已生成的总token数（用于动态调整单轮长度）
        generated_so_far = self.stats['generated_tokens']
        
        # 动态限制单轮最大长度：基础长度 + 已生成长度/100（避免过长单轮）

        max_single_draft = min(
            self.max_draft_len * 2,
            self.max_draft_len + generated_so_far // 100
        )
        max_single_draft = max(max_single_draft, 1)
        
        for i in range(max_single_draft):
            try:
                with torch.no_grad():
                    outputs = self.draft_model(
                        input_ids=current_input,
                        attention_mask=current_mask,
                        return_dict=True
                    )
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token_probs = self._top_k_sampling(next_token_logits)
                next_token = self._sample_token(next_token_probs)
                
                # 草稿阶段检查EOS：若生成EOS，提前终止草稿生成
                if self._check_early_stop(next_token):

                    break
                
                draft_probs_list.append(next_token_probs)
                draft_tokens.append(next_token)
                
                # 更新输入（过滤特殊token，避免污染后续生成）
                current_input = torch.cat([current_input, next_token], dim=1)
                current_mask = torch.cat([
                    current_mask, 
                    torch.ones((batch_size, 1), device=self.device)
                ], dim=1)
                
            except Exception as e:
                self.logger.error(f"草稿生成第{i+1}步失败: {e}")
                break
        
        # 堆叠草稿token（空序列处理）
        draft_sequence = torch.cat(draft_tokens, dim=1) if draft_tokens else torch.tensor([], device=self.device, dtype=torch.long)
        return draft_sequence, draft_probs_list

    def _verify_draft_tokens(self,
                           input_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           draft_sequence: torch.Tensor,
                           draft_probs_list: List[torch.Tensor]) -> Tuple[torch.Tensor, int, bool]:

        self.stats['main_calls'] += 1
        input_ids = self._ensure_npu_tensor(input_ids)
        attention_mask = self._ensure_npu_tensor(attention_mask)
        draft_sequence = self._ensure_npu_tensor(draft_sequence)
        
        batch_size = input_ids.shape[0]
        draft_length = draft_sequence.shape[1] if draft_sequence.numel() > 0 else 0
        early_stop = False
        
        if draft_length == 0:
            # 无草稿token：主模型直接生成，同时检查EOS
            outputs = self.main_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = self._top_k_sampling(next_token_logits)
            next_token = self._sample_token(next_token_probs)
            
            # 检查是否为EOS：若是，触发提前结束
            if self._check_early_stop(next_token):
                early_stop = True
                self.stats['early_stop_count'] += 1
            
            return next_token, 1, early_stop
        
        # 拼接前缀与草稿序列，主模型验证
        verify_input_ids = torch.cat([input_ids, draft_sequence], dim=1)
        verify_attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, draft_length), device=self.device)
        ], dim=1)
        
        with torch.no_grad():
            try:
                main_outputs = self.main_model(
                    input_ids=verify_input_ids,
                    attention_mask=verify_attention_mask,
                    return_dict=True
                )
                main_logits = main_outputs.logits[:, -draft_length-1:-1, :]
                
                accepted_tokens = 0
                final_tokens = []
                
                for i in range(draft_length):
                    draft_token = draft_sequence[:, i]
                    main_probs = self._top_k_sampling(main_logits[:, i, :])
                    
                    # 检查草稿token是否可接受
                    batch_indices = torch.arange(batch_size, device=self.device)
                    draft_token_main_prob = main_probs[batch_indices, draft_token]
                    acceptance_mask = draft_token_main_prob > self.acceptance_threshold
                    
                    if acceptance_mask.all():
                        # 接受草稿token，检查是否为EOS
                        if self._check_early_stop(draft_token):
                            early_stop = True
                            self.stats['early_stop_count'] += 1
                            final_tokens.append(draft_token.unsqueeze(1))
                            accepted_tokens += 1
                            break  # EOS后终止验证
                        final_tokens.append(draft_token.unsqueeze(1))
                        accepted_tokens += 1
                    else:
                        # 拒绝草稿token，重新采样并检查EOS
                        self.stats['rejection_events'] += 1
                        resampled_token = self._sample_token(main_probs)
                        if self._check_early_stop(resampled_token):
                            early_stop = True
                            self.stats['early_stop_count'] += 1
                        final_tokens.append(resampled_token)
                        accepted_tokens += 1
                        break  # 拒绝后终止后续验证
                
                # 所有草稿token被接受：采样下一个token并检查EOS
                if accepted_tokens == draft_length and not early_stop:
                    next_token_logits = main_outputs.logits[:, -1, :]
                    next_token_probs = self._top_k_sampling(next_token_logits)
                    next_token = self._sample_token(next_token_probs)
                    
                    if self._check_early_stop(next_token):
                        early_stop = True
                        self.stats['early_stop_count'] += 1
                    
                    final_tokens.append(next_token)
                    accepted_tokens += 1
                
                final_sequence = torch.cat(final_tokens, dim=1) if final_tokens else torch.tensor([], device=self.device, dtype=torch.long)
                
            except Exception as e:
                self.logger.error(f"大模型验证失败: {e}")
                # 回退策略：取草稿第一个token，检查EOS
                final_sequence = draft_sequence[:, :1] if draft_length > 0 else torch.tensor([], device=self.device, dtype=torch.long)
                accepted_tokens = 1 if draft_length > 0 else 0
                if accepted_tokens > 0 and self._check_early_stop(final_sequence):
                    early_stop = True
                    self.stats['early_stop_count'] += 1
        
        return final_sequence, accepted_tokens, early_stop

    def speculative_decode(self,
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          max_new_tokens: int) -> Tuple[torch.Tensor, Dict]:

        start_time = time.time()
        input_ids = self._ensure_npu_tensor(input_ids)
        attention_mask = self._ensure_npu_tensor(attention_mask)
        
        batch_size = input_ids.shape[0]
        all_generated_tokens = torch.tensor([], device=self.device, dtype=torch.long)
        current_input = input_ids
        current_mask = attention_mask
        
        self.logger.info(f"开始投机推理，最大生成token数={max_new_tokens}（支持EOS提前结束）...")
        
        try:
            generated_count = 0
            iteration = 0
            early_stop = False
            
            while generated_count < max_new_tokens and not early_stop:
                iteration += 1
                
                # 1. 生成草稿序列
                draft_sequence, draft_probs_list = self._generate_draft_sequence(
                    current_input, current_mask
                )
                
                # 2. 验证草稿并检查提前结束
                final_sequence, accepted_count, early_stop = self._verify_draft_tokens(
                    current_input, current_mask, draft_sequence, draft_probs_list
                )
                
                # 3. 过滤最终序列中的特殊token
                final_sequence = self._filter_special_tokens(final_sequence)
                actual_new_tokens = final_sequence.shape[1] if final_sequence.numel() > 0 else 0
                
                # 4. 更新生成计数（避免超过最大token数）
                if generated_count + actual_new_tokens > max_new_tokens:
                    actual_new_tokens = max_new_tokens - generated_count
                    final_sequence = final_sequence[:, :actual_new_tokens]
                
                # 5. 累积生成token
                if actual_new_tokens > 0:
                    all_generated_tokens = torch.cat([all_generated_tokens, final_sequence], dim=1)
                    generated_count += actual_new_tokens
                
                # 6. 更新统计
                self.stats['total_tokens'] += accepted_count
                self.stats['accepted_tokens'] += (accepted_count - 1) if accepted_count > 0 else 0
                self.stats['generated_tokens'] = generated_count
                
                # 7. 更新输入序列
                current_input = torch.cat([input_ids, all_generated_tokens], dim=1)
                current_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, all_generated_tokens.shape[1]), device=self.device)
                ], dim=1)
                
                # 8. 打印进度
                if iteration % 5 == 0 or generated_count >= max_new_tokens or early_stop:
                    acceptance_rate = self.stats['accepted_tokens'] / max(1, self.stats['total_tokens'])
                    stop_msg = "（提前结束）" if early_stop else ""
                    self.logger.info(f"迭代 {iteration}: 已生成 {generated_count}/{max_new_tokens} tokens {stop_msg}, 接受率: {acceptance_rate:.3f}")
                
                # 安全检查：防止无限循环
                if iteration > max_new_tokens * 2:
                    self.logger.warning(f"警告：达到最大迭代次数 {iteration}，提前结束")
                    break
                    
        except Exception as e:
            self.logger.error(f"推理过程中出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 最终过滤（确保无特殊token残留，保留EOS）
        final_generated = self._filter_special_tokens(all_generated_tokens[:, :max_new_tokens])
        
        # 计算性能统计
        total_time = time.time() - start_time
        self.stats['total_time'] = total_time
        tokens_per_sec = final_generated.shape[1] / total_time if total_time > 0 else 0
        acceptance_rate = self.stats['accepted_tokens'] / max(1, self.stats['total_tokens'])
        
        performance_stats = {
            'total_generated_tokens': final_generated.shape[1],
            'total_tokens': self.stats['total_tokens'],
            'accepted_tokens': self.stats['accepted_tokens'],
            'draft_calls': self.stats['draft_calls'],
            'main_calls': self.stats['main_calls'],
            'rejection_events': self.stats['rejection_events'],
            'early_stop_count': self.stats['early_stop_count'],
            'total_iterations': iteration,
            'total_time': total_time,
            'tokens_per_sec': tokens_per_sec,
            'acceptance_rate': acceptance_rate,
            'speedup_ratio': self.stats['total_tokens'] / max(1, self.stats['main_calls'])
        }
        
        stop_summary = f"（因EOS提前结束，实际生成{final_generated.shape[1]}个token）" if self.stats['early_stop_count'] > 0 else ""
        self.logger.info(f"推理完成！目标生成{max_new_tokens}个token {stop_summary}，耗时 {total_time:.2f} 秒")
        return final_generated, performance_stats

    def _filter_special_tokens(self, tokens: torch.Tensor) -> torch.Tensor:

        if tokens.numel() == 0:
            return tokens
        
        # 将token转换为numpy数组，便于过滤
        tokens_np = tokens.cpu().numpy()
        # 保留EOS和非特殊token
        filtered_np = [
            token for token in tokens_np[0] 
            if token == self.tokenizer.eos_token_id or token not in self.special_tokens_to_filter
        ]
        # 转换回tensor（保持batch维度）
        return torch.tensor([filtered_np], device=self.device, dtype=torch.long)

    def _warmup(self):

        self.logger.info(f"进行 {self.device} 模型预热...")
        warmup_input = torch.randint(0, self.tokenizer.vocab_size, (1, 10), device=self.device, dtype=torch.long)
        warmup_mask = torch.ones((1, 10), device=self.device)
        
        with torch.no_grad():
            try:
                _ = self.draft_model(warmup_input, attention_mask=warmup_mask)
                _ = self.main_model(warmup_input, attention_mask=warmup_mask)
                if torch.npu.is_available():
                    torch.npu.synchronize(self.device)
                elif torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
            except Exception as e:
                self.logger.error(f"预热失败: {e}")
        self.logger.info("预热完成")

class SpeculativeCEvalEvaluator:
    def __init__(self, main_model_path, draft_model_path, data_dir):
        self.output_dir = "./7bresult_speculative"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化日志
        self.logger = setup_logger(self.output_dir)
        
        # 打印并记录环境和模型信息
        self._log_system_info(main_model_path, draft_model_path)

        # 加载模型和tokenizer
        # 统一使用配置好的device
        self.device = torch.device("npu:4" if torch.npu.is_available() else "cpu")
        self.logger.info(f"初始化 Evaluator，使用设备: {self.device}")
        
        # 加载主模型和草稿模型
        self.main_model = AutoModelForCausalLM.from_pretrained(
            main_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(main_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 初始化投机解码器 - 传递 device 和 logger
        self.decoder = TopKSpeculativeDecoder(
            main_model=self.main_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            device=self.device,
            k=5,
            max_draft_len=3,
            temperature=0.8,
            top_p=0.9,
            acceptance_threshold=0.01,
            logger=self.logger
        )
        
        self.data_dir = data_dir
        self.results = []
        self.total_stats = {
            'total_questions': 0,
            'total_tokens': 0,
            'total_generated_tokens': 0,
            'total_time': 0,
            'all_stats': []
        }
    
    def _log_system_info(self, main_path, draft_path):

        self.logger.info("=" * 60)
        self.logger.info("SYSTEM AND MODEL CONFIGURATION")
        self.logger.info("=" * 60)
        
        # 模型信息
        self.logger.info(f"Main Model Path: {main_path}")
        self.logger.info(f"Draft Model Path: {draft_path}")
        
        # 硬件信息
        if torch.npu.is_available():
            try:
                device_count = torch.npu.device_count()
                current_device_id = torch.npu.current_device()
                device_name = torch.npu.get_device_name(current_device_id)
                device_props = torch.npu.get_device_properties(current_device_id)
                total_memory_gb = device_props.total_memory / (1024**3)
                
                self.logger.info(f"Platform: Huawei Ascend NPU")
                self.logger.info(f"NPU Device Count: {device_count}")
                self.logger.info(f"Current Device ID: {current_device_id}")
                self.logger.info(f"Device Name: {device_name}")
                self.logger.info(f"Total Video Memory: {total_memory_gb:.2f} GB")
                
            except Exception as e:
                self.logger.error(f"Failed to retrieve NPU info: {e}")
        else:
            self.logger.info("Platform: CPU")
        
        self.logger.info("=" * 60)

    def build_prompt(self, question, options):
        prompt = f"请回答以下选择题：\n\n{question}\n"
        for option, text in options.items():
            prompt += f"{option}. {text}\n"
        prompt += "\n请选择正确的选项："
        return prompt
    
    def find_csv_files(self):
        pattern = os.path.join(self.data_dir, "**", "*.csv")
        csv_files = glob.glob(pattern, recursive=True)
        return sorted(csv_files)
    
    def process_csv_file(self, csv_file):

        self.logger.info(f"处理文件: {csv_file}")
        file_results = []
        
        try:
            df = pd.read_csv(csv_file)
            
            for index, row in df.iterrows():
                # 检查列是否存在
                if len(row) < 7:
                    self.logger.warning(f"警告: 第{index+1}行列数不足，跳过")
                    continue
                
                # 构建选项字典
                options = {
                    'A': row[2] if len(row) > 2 and pd.notna(row[2]) else '',
                    'B': row[3] if len(row) > 3 and pd.notna(row[3]) else '',
                    'C': row[4] if len(row) > 4 and pd.notna(row[4]) else '',
                    'D': row[5] if len(row) > 5 and pd.notna(row[5]) else ''
                }
                
                # 过滤空选项
                options = {k: v for k, v in options.items() if v}
                
                if not options:
                    self.logger.warning(f"警告: 第{index+1}行没有有效选项，跳过")
                    continue
                
                # 构建提示词
                prompt = self.build_prompt(row[1], options)
                
                # 使用投机推理生成回答
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                # 预热模型
                if index == 0:
                    self.decoder._warmup()
                
                # 使用投机推理生成回答
                generated_tokens, stats = self.decoder.speculative_decode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2048
                )
                
                # 解码回复
                response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                # 保存结果
                result = {
                    'file': os.path.basename(csv_file),
                    'id': row[0],
                    'question': row[1],
                    'options': options,
                    'correct_answer': row[6] if len(row) > 6 else '',
                    'model_response': response,
                    'stats': stats
                }
                
                file_results.append(result)
                
                # 更新总统计
                self.total_stats['total_questions'] += 1
                self.total_stats['total_tokens'] += stats['total_tokens']
                self.total_stats['total_generated_tokens'] += stats['total_generated_tokens']
                self.total_stats['total_time'] += stats['total_time']
                self.total_stats['all_stats'].append(stats)
                
                self.logger.info(f"进度: {index+1}/{len(df)} | Tokens: {stats['total_generated_tokens']} | Speed: {stats['tokens_per_sec']:.2f} tokens/s")
            
            return file_results
            
        except Exception as e:
            self.logger.error(f"处理文件 {csv_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return file_results
    
    def save_file_results(self, file_results, csv_file, file_index, total_files):

        if not file_results:
            self.logger.info(f"文件 {csv_file} 没有有效结果，跳过保存")
            return
        
        filename = os.path.basename(csv_file)
        base_name = os.path.splitext(filename)[0]
        
        # 保存详细结果
        output_file = os.path.join(self.output_dir, f"{base_name}_speculative_results.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"CEval投机推理评估结果 - {filename}\n")
            f.write(f"处理进度: {file_index}/{total_files} 个文件\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(file_results):
                f.write(f"问题 {i+1}:\n")
                f.write(f"ID: {result['id']}\n")
                f.write(f"问题: {result['question']}\n")
                f.write("选项:\n")
                for option, text in result['options'].items():
                    f.write(f"  {option}. {text}\n")
                f.write(f"正确答案: {result['correct_answer']}\n")
                f.write(f"模型回答: {result['model_response']}\n")
                
                stats = result['stats']
                f.write(f"投机推理统计:\n")
                f.write(f"  生成token数: {stats['total_generated_tokens']}\n")
                f.write(f"  总token数: {stats['total_tokens']}\n")
                f.write(f"  接受率: {stats['acceptance_rate']:.4f}\n")
                f.write(f"  草稿调用: {stats['draft_calls']}\n")
                f.write(f"  主模型调用: {stats['main_calls']}\n")
                f.write(f"  加速比: {stats['speedup_ratio']:.2f}\n")
                f.write(f"  生成速度: {stats['tokens_per_sec']:.2f} tokens/s\n")
                f.write(f"  总时间: {stats['total_time']:.2f}秒\n")
                f.write("-" * 80 + "\n\n")
        
        self.logger.info(f"文件结果已保存到: {output_file}")
        
        # 保存当前文件的统计信息
        self.save_current_stats(file_index, total_files)
    
    def save_current_stats(self, current_file_index, total_files):

        stats_file = os.path.join(self.output_dir, f"progress_stats_{current_file_index:03d}.txt")
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("CEval投机推理评估进度统计\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"处理进度: {current_file_index}/{total_files} 个文件\n")
            f.write(f"已处理问题数: {self.total_stats['total_questions']}\n")
            f.write(f"总token数: {self.total_stats['total_tokens']}\n")
            f.write(f"总生成token数: {self.total_stats['total_generated_tokens']}\n")
            f.write(f"总生成时间: {self.total_stats['total_time']:.2f}秒\n")
            
            if self.total_stats['total_questions'] > 0:
                avg_tokens_per_question = self.total_stats['total_generated_tokens'] / self.total_stats['total_questions']
                avg_time_per_question = self.total_stats['total_time'] / self.total_stats['total_questions']
                overall_speed = self.total_stats['total_generated_tokens'] / self.total_stats['total_time'] if self.total_stats['total_time'] > 0 else 0
                
                f.write(f"平均每问题生成token数: {avg_tokens_per_question:.2f}\n")
                f.write(f"平均每问题生成时间: {avg_time_per_question:.2f}秒\n")
                f.write(f"总体生成速度: {overall_speed:.2f} tokens/s\n")
    
    def save_final_summary(self, total_files):

        summary_file = os.path.join(self.output_dir, "final_speculative_summary.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("CEval投机推理评估最终汇总报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"总文件数: {total_files}\n")
            f.write(f"总问题数: {self.total_stats['total_questions']}\n")
            f.write(f"总token数: {self.total_stats['total_tokens']}\n")
            f.write(f"总生成token数: {self.total_stats['total_generated_tokens']}\n")
            f.write(f"总生成时间: {self.total_stats['total_time']:.2f}秒\n")
            
            if self.total_stats['total_questions'] > 0:
                avg_tokens_per_question = self.total_stats['total_generated_tokens'] / self.total_stats['total_questions']
                avg_time_per_question = self.total_stats['total_time'] / self.total_stats['total_questions']
                overall_speed = self.total_stats['total_generated_tokens'] / self.total_stats['total_time'] if self.total_stats['total_time'] > 0 else 0
                
                f.write(f"平均每问题生成token数: {avg_tokens_per_question:.2f}\n")
                f.write(f"平均每问题生成时间: {avg_time_per_question:.2f}秒\n")
                f.write(f"总体生成速度: {overall_speed:.2f} tokens/s\n")
            
            # 投机推理特有统计
            if self.total_stats['all_stats']:
                avg_acceptance_rate = sum(s['acceptance_rate'] for s in self.total_stats['all_stats']) / len(self.total_stats['all_stats'])
                avg_speedup = sum(s['speedup_ratio'] for s in self.total_stats['all_stats']) / len(self.total_stats['all_stats'])
                f.write(f"平均接受率: {avg_acceptance_rate:.4f}\n")
                f.write(f"平均加速比: {avg_speedup:.2f}\n")
            
            # 按文件统计
            f.write("\n各文件处理统计:\n")
            file_stats = {}
            for result in self.results:
                filename = result['file']
                if filename not in file_stats:
                    file_stats[filename] = {'questions': 0, 'tokens': 0}
                file_stats[filename]['questions'] += 1
                file_stats[filename]['tokens'] += result['stats']['total_generated_tokens']
            
            for filename, stats in file_stats.items():
                f.write(f"{filename}: {stats['questions']} 问题, {stats['tokens']} tokens\n")
        
        self.logger.info(f"最终汇总报告已保存到: {summary_file}")
    
    def evaluate_all(self):
        
        csv_files = self.find_csv_files()
        total_files = len(csv_files)
        self.logger.info(f"找到 {total_files} 个CSV文件")
        
        if not csv_files:
            self.logger.warning("未找到CSV文件，请检查路径")
            return
        
        self.logger.info(f"结果将保存到目录: {self.output_dir}")
        
        for file_index, csv_file in enumerate(csv_files, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"处理第 {file_index}/{total_files} 个文件: {os.path.basename(csv_file)}")
            self.logger.info(f"{'='*60}")
            
            # 处理单个文件
            file_results = self.process_csv_file(csv_file)
            self.results.extend(file_results)
            
            # 立即保存该文件的结果
            self.save_file_results(file_results, csv_file, file_index, total_files)
            
            self.logger.info(f"第 {file_index} 个文件处理完成")
        
        # 所有文件处理完成后保存最终汇总
        self.save_final_summary(total_files)
        
        self.logger.info(f"\n所有文件处理完成！")
        self.logger.info(f"详细结果请查看目录: {self.output_dir}")
        
        return self.results

def main():
    # 模型路径配置
    main_model_path = "/opt/pangu/openPangu-Embedded-7B-V1.1"
    draft_model_path = "/opt/pangu/openPangu-Embedded-1B-V1.1"
    data_dir = "/opt/pangu/zc/evaldemo"
    
    try:
        print("开始CEval投机推理评估...")
        evaluator = SpeculativeCEvalEvaluator(main_model_path, draft_model_path, data_dir)
        results = evaluator.evaluate_all()
        
        print("投机推理评估完成！")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
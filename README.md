# An Adaptive Optimization Framework for Pangu Large Models

[中文](README.zh-CN.md) | **English**

This repository provides a full-stack optimization solution for the **Pangu-Embedded** series large models (1B/7B) on the **Huawei Ascend NPU** platform. The solution covers weight compression (W8A16 quantization), structured pruning (Taylor iterative pruning + distillation), speculative inference (Draft Model, PLD), and an adaptive intent recognition framework.

---

## Core Features

-   **W8A16 Mixed-Precision Quantization**: Int8 weight storage with BF16 runtime de-quantization calculation.
-   **Taylor-based Iterative Pruning**: Utilizes first-order gradient importance scoring combined with 256-bit NPU hardware alignment to ensure physical acceleration.
-   **Knowledge Distillation**: Recovers model accuracy at high pruning rates via KL divergence.
-   **Dual-Mode Speculative Inference**:
    -   **Draft Model Scheme**: 7B main model + 1B draft model collaboration to enhance logical consistency.
    -   **PLD (Pattern Lookahead Decoding)**: Zero-overhead draft generation based on N-Gram matching, achieving up to 2x speedup.
-   **Adaptive Optimization Framework**: Uses the 1B model to automatically identify user intent and intelligently recommend the most matching inference strategy.

---

## Technical Details

### 1. W8A16 Mixed-Precision Quantization
Implements a "low-storage, high-precision" quantization scheme.
Linear layer weights are quantized to Int8. During forward propagation, weights are restored to BF16 in real-time for matrix multiplication with activations.
-   **Algorithm**: Calculates per-channel scaling factor $Scale = \frac{\max(|W|)}{127}$, mapping weights to $[-128, 127]$.
-   **Memory Benefit**: Reduces Pangu-1B memory footprint from **2.15GB** to **1.34GB**.

### 2. Iterative Taylor Pruning & Multi-dimensional Distillation
Performs structured removal on MLP dimensions using a 5-round linear schedule.
-   **Taylor Scoring**: $Score = |W \cdot \nabla W|$, accurately identifying neurons with the least contribution to loss function fluctuation.
-   **NPU Alignment**: Forces retained dimensions to be multiples of 256 to eliminate physical hardware padding and maximize compute utilization.
-   **Optimal Config**: 5e-5 Learning Rate + Linear Schedule + KL Distillation.

### 3. Speculative Inference Acceleration
Addresses the serial bottleneck of auto-regressive generation with two acceleration schemes:
-   **Draft Model Scheme**: Uses the 1B model as a "Draft" to pre-generate $K$ tokens, while the 7B "Target" model performs parallel verification, significantly increasing throughput.
-   **PLD Scheme**: Pattern Lookahead Decoding based on N-Gram matching. Utilizes `NGramMatcher` to retrieve repeating patterns in historical context, achieving zero-parameter draft generation.

### 4. Adaptive Optimization Framework
Integrates intent recognition logic. It uses the 1B model to determine the user's task type and automatically recommends the optimal combination of quantization, pruning, and acceleration strategies based on benchmarks.

---

## Experimental Results

### Quantization Performance (Code Generation)

| Model State | MBPP Score |
| :--- | :--- |
| Original | 54.09 |
| Quantized | 53.31 |

### Pruning & Distillation Performance (Math Reasoning)

| MLP Pruning Rate | GSM8K Score |
| :--- | :--- |
| 4.17% | 71.87 |
| 8.33% | 69.52 |
| 12.5% | 64.59 |
| 16.67% | 61.87 |
| 20.83% | 61.87 |

### Speculative Inference Speedup (Tokens/s)
*Example based on Pangu-1B using PLD Scheme (N=3, K=4):*

| Model | Task | Baseline Speed | PLD Speed | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Pangu-1B | C-Eval | 21.58 | 41.93 | **1.94x** |
| Pangu-1B | MMLU | 21.87 | 43.41 | **1.98x** |

---

## Project Structure

-   `src/main_distill_prune.py`: Core implementation of pruning and knowledge distillation.
-   `src/main_quantization.py`: Definitions for W8A16 quantizer and de-quantized linear layers.
-   `src/main_speculative_draft_ceval.py`: Large-Small model speculative decoder based on Top-K sampling.
-   `src/main_speculative_pld.py`: Pattern Lookahead Decoding implementation based on N-Gram matching (Adapted for OpenCompass).
-   `src/model_framework.py`: Framework integration utilities.
-   `docs/`: Detailed experimental reports and technical analysis documents.

---

## Requirements

-   **Hardware**: Huawei Ascend 910B/310P NPU
-   **Framework**: `torch >= 2.1.0`, `torch_npu`, `transformers >= 4.34.0`
-   **CANN**: Huawei Ascend CANN (Community or Commercial Edition)

---

## Citation & Acknowledgements

This research is based on the Pangu Large Model series. We thank Huawei Ascend for providing computing resources and `torch_npu` technical support.

---

**Maintainer**: [Institute of Intelligent Computing Systems (IICS), Hunan University]

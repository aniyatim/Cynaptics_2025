# README: Fine-Tuning Falcon-7B-Instruct for Persona-Chat Dataset

## 1. **Choice of LLM**

### Model: `vilsonrodrigues/falcon-7b-instruct-sharded`

- **Instruction Tuned:** Falcon-7B-Instruct is fine-tuned on instruction-based datasets, making it a strong foundation for generating context-aware responses.
- **Parameters:** At 7 billion parameters, it provides a balance between computational efficiency and high performance, making it suitable for fine-tuning on consumer-grade GPUs.
- **Quantization:** Supports 4-bit quantization, enabling memory-efficient fine-tuning.

---

## 2. **Choice of Fine-Tuning Method**

### Fine-Tuning Method: **LoRA (Low-Rank Adaptation)**

1. **Parameter-Efficient Fine-Tuning:**
   - LoRA allows updating only a small subset of model parameters, significantly reducing computational requirements.
2. **Scalability:**
   - LoRA is compatible with large-scale models like Falcon-7B, enabling efficient adaptation without retraining the entire model.

---

## 3. **Link to Model Weights**

The fine-tuned model weights have been uploaded to a public Hugging Face repository:

[**Falcon-7B Persona-Chat Fine-Tuned Model**](https://huggingface.co/niyatimishra/fine_tuned_falcon7b_dialogue)

---

## 4. **Visualizations**

1. **Training Loss Curve:** Demonstrates the reduction in loss over training steps.
2. **Gradient Norms:** Ensures gradients remained stable throughout training.
3. **Learning Rate Schedule:** Highlights the cosine decay schedule used.

These graphs can be found in the `Images/` folder of this repository.

---


# -*- coding: utf-8 -*-
"""myLLM

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/myllm-261de185-24e6-471b-ba4c-b934be6f1195.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20250117/auto/storage/goog4_request%26X-Goog-Date%3D20250117T124854Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D445d598a116ab2548baa3e2a3656aedcb7d3991cbc8b21cce49127cbc87915d22c2001a006b6dd1f70ba09d4a9eeb4a73e3f38ccbe17adbfe9a94f38e3a7d3efe6b5e0e9a8d03a01a4211298cee0e6d55e6a715661dcef7322906210e35c26968498d3a0b014388ef7a8a8c4be299b272cd1d1b9f7e8cebcba51e8525873b96662703e00c344d23a4ba9fc58a8e89fd2a8d978e10531e3f4be2c6a924a1d7fabe5f750c10f7bddc115fe67ff957812024cca9b5660c4ac59ec0690f453fcdcc56a9bab60d0807493d2c3cb1973831f4ee66de1efa0f736e61cc6be09bed90478c28f7013e34ffa095d2982104c0c84c713c3a4218085710c486f799675c88ac1
"""

!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
!pip install -q datasets bitsandbytes einops wandb
!pip install huggingface_hub

import torch
import time
from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

import warnings
warnings.filterwarnings("ignore")

from huggingface_hub import notebook_login
notebook_login()

dataset_name = "Cynaptics/persona-chat"
data = load_dataset(dataset_name)
data

model_name = "vilsonrodrigues/falcon-7b-instruct-sharded"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

#LoRA
model = prepare_model_for_kbit_training(model)

lora_alpha = 32
lora_dropout = 0.05
lora_rank = 32

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_rank,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
)

peft_model = get_peft_model(model, peft_config)

#
output_dir = "fine_tuned_falcon7b_dialogue"
per_device_train_batch_size = 16
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 180
warmup_ratio = 0.03
lr_scheduler_type = "cosine"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    push_to_hub=True
)

#preprocess
from datasets import load_dataset, Dataset
dataset = load_dataset("Cynaptics/persona-chat")

def preprocess(row):

    persona_b = " ".join(row["persona_b"])
    dialogue = " ".join(row["dialogue"])

    prompt = f"Persona: {persona_b}"
    instruction = f"Dialogue: {dialogue}"

    return {
        "text": f"{prompt}\n\n{instruction}\n\n{row['reference']}"
    }

processed_data = [preprocess(row) for row in dataset["train"]]

train_dataset = Dataset.from_list(processed_data)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

processed_data = [preprocess(row) for row in dataset["train"]]

from datasets import Dataset
train_dataset = Dataset.from_list(processed_data)

tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

train_dataset

train_dataset['text'][:4]

#SFTTrainer
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
)

#stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.bfloat16)

from accelerate import Accelerator
accelerator = Accelerator()

import wandb
wandb.login(key="api_key")

peft_model.config.use_cache = False
trainer.train()

trainer.push_to_hub()

#loading model
PEFT_MODEL = "niyatimishra/fine_tuned_falcon7b_dialogue"
config = PeftConfig.from_pretrained(PEFT_MODEL)
peft_base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

peft_model = PeftModel.from_pretrained(peft_base_model, PEFT_MODEL)

peft_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
peft_tokenizer.pad_token = peft_tokenizer.eos_token

prompt = f"""
Person B has the following Persona information.

Persona of Person B: My name is David and I'm a 35 year old math teacher.
Persona of Person B: I like to hike and spend time in the nature.
Persona of Person B: I'm married with two kids.

Instruct: Person A and Person B are now having a conversation.
Following the conversation below, write a response that Person B would say based on the
above Persona information.
Please carefully consider the flow and context of the conversation below, and use the Person B's Persona information appropriately to generate a response that you think is
the most appropriate reply for Person B.

Persona A: Morning! I think I saw you at the parent meeting, what is your name?

Output:
"""

#tokenize
inputs = peft_tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    padding=True
)

input_ids = inputs["input_ids"].cuda()
attention_mask = inputs["attention_mask"].cuda()
#print("hello 2")

#inference
with torch.inference_mode():
    outputs = peft_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.1,
        temperature=0.7,
        pad_token_id=peft_tokenizer.pad_token_id,
    )
#print("hello 1")

decoded_outputs = peft_tokenizer.batch_decode(outputs, skip_special_tokens=True)
output = decoded_outputs[0][len(prompt):]
print("Generated Output:\n", output)
#print("hello")

prompt = f"""
Person B has the following Persona information.

Persona of Person B: My name is Sarah, and I'm a 28-year-old software developer.
Persona of Person B: I love reading science fiction novels and exploring new technologies.
Persona of Person B: I recently adopted a rescue dog named Luna.

Instruct: Person A and Person B are now having a conversation.
Following the conversation below, write a response that Person B would say based on the
above Persona information.
Please carefully consider the flow and context of the conversation below, and use Person B's Persona information appropriately to generate a response that you think is
the most appropriate reply for Person B.

Persona A: Hey, Sarah! What’s been keeping you busy these days?

Output:
"""

inputs = peft_tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    padding=True
)

input_ids = inputs["input_ids"].cuda()
attention_mask = inputs["attention_mask"].cuda()
#print("hello 2")


with torch.inference_mode():
    outputs = peft_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.1,
        temperature=0.7,
        pad_token_id=peft_tokenizer.pad_token_id,
    )
#print("hello 1")

decoded_outputs = peft_tokenizer.batch_decode(outputs, skip_special_tokens=True)
output = decoded_outputs[0][len(prompt):]
print("Generated Output:\n", output)
#print("hello")

#prompt function
def response(persona,dialogue):
    prompt = f"""

        f{persona}

        Instruct: Person A and Person B are now having a conversation.
        Following the conversation below, write a response that Person B would say based on the
        above Persona information.
        Please carefully consider the flow and context of the conversation below, and use Person B's Persona information appropriately to generate a response that you think is
        the most appropriate reply for Person B.

        f{dialogue}
    """

    inputs = peft_tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    padding=True
    )

    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    #print("hello 2")

    with torch.inference_mode():
        outputs = peft_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.1,
        temperature=0.7,
        pad_token_id=peft_tokenizer.pad_token_id,
        )
    #print("hello 1")

    decoded_outputs = peft_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #print("hello")

    output = decoded_outputs[0][len(prompt):]
    print(output)

persona = """
Persona of Person B: My name is Sarah, and I'm a 28-year-old software developer.
Persona of Person B: I love reading science fiction novels and exploring new technologies.
Persona of Person B: I recently adopted a rescue dog named Luna.
"""
dialogue = "Persona A: Hey, Sarah! how is your dog ? "
response(persona,dialogue)

persona = """
Persona of Person B: My name is Alex, and I'm a 32-year-old graphic designer.
Persona of Person B: I enjoy painting landscapes and playing the guitar in my free time.
Persona of Person B: I recently started taking cooking classes to explore new cuisines.
"""

dialogue = "Persona A: Hey Alex! I heard you’re learning to cook. What’s the most exciting dish you’ve tried so far?"

response(persona,dialogue)

persona = """
Persona of Person B: My name is Emma, and I'm a 25-year-old yoga instructor.
Persona of Person B: I love spending time at the beach and practicing mindfulness.
Persona of Person B: I’m currently planning a trip to Bali for a yoga retreat.
"""

dialogue = "Persona A: Hi Emma! I saw your post about Bali. What inspired you to plan a yoga retreat there?"


response(persona,dialogue)
import torch
from unsloth import FastLanguageModel
import json
import glob
from datasets import Dataset
from datasets import DatasetDict
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import evaluate

max_seq_length = 1024
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="davidkim205/ko-gemma-2-9b-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

prompt = """<start_of_turn>user
다음 문장을 점역해줘.
{}
<end_of_turn><start_of_turn>model
{}"""


def formatting_prompts_func(examples):
    inputs = examples["source"]
    outputs = examples["target"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = prompt.format(input, output)
        texts.append(text + "<end_of_turn><eos>")
    return {"text": texts, }


def load_data():
    src_texts = []
    tgt_texts = []

    json_files = glob.glob('./data/*.json')

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data.get('parallel', []):
                src_texts.append(item['source'])
                tgt_texts.append(item['target'])

    dataset = Dataset.from_dict({"source": src_texts, "target": tgt_texts})

    return dataset


data = load_data()

train_test_split = data.train_test_split(train_size=0.7, test_size=0.3, shuffle=True)
eval_test_split = train_test_split['test'].train_test_split(train_size=0.5, test_size=0.5, shuffle=True)

dataset_split = DatasetDict({
    'train': train_test_split['train'],
    'eval': eval_test_split['train'],
    'test': eval_test_split['test']
})

dataset_split['train'] = dataset_split['train'].map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_split['train'],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer_stats = trainer.train()

FastLanguageModel.for_inference(model)
text_ref = []
text_gen = []
for i in dataset_split['test']:
    input_text = prompt.format(
        i['source'],
        "",
    )
    inputs = tokenizer(
        [
            input_text
        ], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    generated = tokenizer.batch_decode(outputs)[0][len(input_text):]
    generated = generated.replace('<end_of_turn><eos>', '')
    text_ref.append(i['target'])
    text_gen.append(generated)

accuracy_metric = evaluate.load("accuracy")
results = accuracy_metric.compute(references=text_ref, predictions=text_gen)
print(results)

#1. 초기 설정 및 모델 로드

# 필요한 라이브러리 임포트
import torchfrom unsloth import FastLanguageModel  #
import json  #
import glob  # 
from datasets import Dataset, DatasetDict  #
from trl import SFTTrainer  #
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import evaluate

# 모델 파라미터 설정
max_seq_length = 1024
dtype = None  # 데이터 타입 설정
load_in_4bit = True  # 4비트 양자화

# 기본 모델 로드
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="davidkim205/ko-gemma-2-9b-it"  # 한국어 Gamma 모델 사용
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

# PEFT(Parameter-Efficient Fine-Tuning) 설정
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # LoRA 랭크
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],  # LoRA 적용될 레이어
    lora_alpha=8,
    lora_dropout=0,
    bias="None",                
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,  #
    loftq_config=None,  #
)


# 2. 데이터 처리 및 포맷팅

# 프롬프트 템플릿 정의
prompt = """<start_of_turn>user
다음 문장을 점역해줘.
{}
<end_of_turn><start_of_turn>model
{}"""

# 프롬프트 포맷팅 함수
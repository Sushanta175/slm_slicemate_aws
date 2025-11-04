# Phi-3-mini 3.8 B  ---------------> 4-bit QLoRA
MODEL_ID      = "microsoft/Phi-3-mini-4k-instruct"
LORA_RANK     = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05
TARGET_MODULES= ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

MAX_LEN       = 1024
PER_DEVICE_BS = 1
GRAD_ACC      = 8          # eff batch 8
EPOCHS        = 1
LR            = 2e-4
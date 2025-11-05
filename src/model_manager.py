# src/model_manager.py
import gc
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Change this to your preferred model / HF id
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Global persistent state
_state = {"tok": None, "model": None, "mode": "try_once"}  # modes: try_once | persistent | percall

def try_load_persistent(device_map="auto", dtype=torch.bfloat16, offload_folder="offload"):
    """
    Try to load tokenizer+model once into GPU (or with auto device_map).
    On failure (OOM or other), cleanup and switch to percall mode.
    Returns True if persistent loaded, False otherwise.
    """
    if _state["mode"] == "persistent" and _state["model"] is not None:
        return True
    try:
        print("model_manager: attempting persistent load:", MODEL)
        bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False)
        tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb,
            device_map=device_map,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            offload_folder=offload_folder,
            offload_buffers=True
        )
        model.eval()
        _state["tok"], _state["model"], _state["mode"] = tok, model, "persistent"
        print("model_manager: persistent load succeeded")
        return True
    except Exception as e:
        print("model_manager: persistent load FAILED, falling back to per-call. Error:")
        traceback.print_exc()
        cleanup_persistent()
        _state["mode"] = "percall"
        return False

def cleanup_persistent():
    try:
        if _state.get("model") is not None:
            try:
                del _state["model"]
            except Exception:
                pass
        if _state.get("tok") is not None:
            try:
                del _state["tok"]
            except Exception:
                pass
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    _state["tok"], _state["model"] = None, None

def load_model_percall(dtype=torch.float16, offload_folder="offload"):
    """
    Load tokenizer+model for a single call. Returns (tok, model).
    Caller must call unload_model_percall(tok, model) after use.
    """
    try:
        bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False)
        tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        # Use auto device map if CUDA available, otherwise CPU-only
        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb,
            device_map=device_map,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            offload_folder=offload_folder,
            offload_buffers=True
        )
        model.eval()
        return tok, model
    except Exception:
        # attempt a safer CPU-only fallback
        try:
            print("model_manager: percall load failed, trying CPU-only fallback")
            tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": "cpu"})
            model.eval()
            return tok, model
        except Exception:
            raise

def unload_model_percall(tok, model):
    try:
        del model
    except Exception:
        pass
    try:
        del tok
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def get_mode():
    return _state["mode"]

def get_persistent():
    return _state["tok"], _state["model"]

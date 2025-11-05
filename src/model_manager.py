# --- replace the try_load_persistent and load_model_percall definitions in src/model_manager.py ---

def try_load_persistent(device_map="auto", dtype=None, offload_folder="offload"):
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

        # NOTE: Do NOT pass torch_dtype when using bitsandbytes quantization_config.
        # The model weights are already placed on devices/dtype by bnb; passing torch_dtype
        # can cause internal `.to(...)` calls which are unsupported for bnb quantized models.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb,
            device_map=device_map,
            low_cpu_mem_usage=True,
            offload_folder=offload_folder,
            offload_buffers=True
        )
        model.eval()
        _state["tok"], _state["model"], _state["mode"] = tok, model, "persistent"
        print("model_manager: persistent load succeeded")
        return True
    except Exception as e:
        print("model_manager: persistent load FAILED, falling back to per-call. Error:")
        # print stack for debugging
        import traceback
        traceback.print_exc()
        cleanup_persistent()
        _state["mode"] = "percall"
        return False


def load_model_percall(dtype=torch.float16, offload_folder="offload"):
    """
    Load tokenizer+model for a single call. Returns (tok, model).
    Caller must call unload_model_percall(tok, model) after use.
    """
    try:
        bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False)
        tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

        # For bitsandbytes quantized load, do not pass torch_dtype here either.
        # Use device_map="auto" so accelerate places shards properly.
        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb,
            device_map=device_map,
            low_cpu_mem_usage=True,
            offload_folder=offload_folder,
            offload_buffers=True
        )
        model.eval()
        return tok, model
    except Exception as e:
        # attempt a safer CPU-only fallback
        print("model_manager: percall load failed (quantized path). Trying CPU-only fallback. Err:", e)
        try:
            tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": "cpu"})
            model.eval()
            return tok, model
        except Exception:
            print("model_manager: CPU-only fallback also failed.")
            raise

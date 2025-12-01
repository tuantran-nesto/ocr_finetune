from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
import torch
import gc
import os
from time import time
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("CUDA is available.")
else:
    print("CUDA is not available.")

# Read HF_TOKEN from environment variable in .env
load_dotenv()
token = os.getenv("HF_TOKEN", "")
login(token=token)
pretrained_model_name = "google/gemma-3n-E2B-it"
pretrained_model_name = "google/gemma-3-4b-it"
# Helper function for inference
def do_gemma_3n_inference(model, messages, max_new_tokens = 2048):
    
    start_time = time()
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    # model = AutoModelForCausalLM.from_pretrained(model, quantization_config=quantization_config)

    # Initialize tokenizer (replace 'your-model-name' with the correct model name)
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path, use_fast = False)

    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    )
    inputs = inputs.to("cuda" if cuda_available else "cpu")
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    response = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = streamer,
    )
    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()
    execution_time = time() - start_time
    print(f"Inference completed in {execution_time:.2f} seconds.")
    return tokenizer.decode(response[0], skip_special_tokens = True)
    


quantization_config = BitsAndBytesConfig()
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name
                                            #  , quantization_config=quantization_config
                                             )

messages = [
    {
        "role": "user", "content": [
            {"type": "text", "text": "Hi, how are you?"}
        ]
    }
]

response = do_gemma_3n_inference(model, messages)
# print(response)
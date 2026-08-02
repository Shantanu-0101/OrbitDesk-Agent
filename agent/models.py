"""
agent/models.py
---------------
Handles the loading and inference for the local text generation model.
We use a small, CPU-friendly model (Qwen 0.5B) to ensure it runs completely locally
without requiring a GPU.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ── Configuration 

# We use Qwen2.5-0.5B-Instruct because it's tiny (~1.5GB), fast on CPU, 
# and very good at following JSON schemas and system prompts.
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Local LLM Class 

class LocalLLM:
    """
    Manages the generation model pipeline.
    """
    
    def __init__(self):
        self.pipeline = None
        self.is_ready = False

    def build(self):
        """
        Loads the tokenizer and model into memory.
        Downloads the model on the first run.
        """
        print("\n" + "="*55)
        print("  GENERATION MODEL SETUP")
        print("="*55)
        
        print(f"\nLoading text generation model: {GENERATION_MODEL_NAME}")
        print("(This will download ~1.5GB on the first run, please be patient...)")
        
        t0 = time.time()
        
        # Determine device (CPU by default on standard laptops)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device.upper()}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL_NAME, 
            torch_dtype=torch.float32 if device == "cpu" else torch.float16
        )
        
        # Create a pipeline for easy text generation
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,       # Max length of the generated answer
            temperature=0.1,          # Low temperature = more factual, less creative hallucination
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        load_time = time.time() - t0
        print(f"\nGeneration model loaded in {load_time:.1f}s")
        self.is_ready = True
        print("\n  LLM is ready!\n" + "="*55 + "\n")

    def generate(self, prompt: str) -> str:
        """
        Generates text based on the provided prompt.
        """
        if not self.is_ready:
            raise RuntimeError("LLM not built yet. Call llm.build() first.")
            
        # The Qwen model uses a specific chat template. We wrap the raw prompt 
        # in the standard user/assistant format for best results.
        messages = [
            {"role": "system", "content": "You are a helpful support agent. Always output valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply the chat template
        formatted_prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Run inference
        outputs = self.pipeline(formatted_prompt)
        
        # Extract just the generated text (ignoring the prompt part)
        generated_text = outputs[0]["generated_text"][len(formatted_prompt):].strip()
        
        return generated_text


# ── Singleton pattern 

_llm_instance: LocalLLM = None

def get_llm() -> LocalLLM:
    """
    Returns the shared LocalLLM instance, building it if needed.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM()
        _llm_instance.build()
    return _llm_instance


# ── Quick test 
if __name__ == "__main__":
    llm = get_llm()
    test_prompt = "Return a JSON object with a single key 'hello' and value 'world'."
    print("\nTesting generation...")
    t0 = time.time()
    response = llm.generate(test_prompt)
    print(f"\nResponse (took {time.time()-t0:.1f}s):\n{response}")
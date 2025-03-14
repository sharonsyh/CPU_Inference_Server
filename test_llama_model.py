from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def test_llama_model(model_name: str, input_text: str):
    # Check if necessary files exist
    if not os.path.exists(os.path.join(model_name, 'config.json')):
        raise FileNotFoundError(f"config.json not found in {model_name}")
    if not os.path.exists(os.path.join(model_name, 'consolidated.00.pth')):
        raise FileNotFoundError(f"Model weights not found in {model_name}")
    if not os.path.exists(os.path.join(model_name, 'tokenizer.model')):
        raise FileNotFoundError(f"Tokenizer model not found in {model_name}")

    # Load the tokenizer and model
    print("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer and model: {e}")

    # Tokenize the input text
    print("Tokenizing input text...")
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate output
    print("Generating output...")
    with torch.no_grad():
        output = model.generate(inputs.input_ids, max_length=50, do_sample=True)

    # Decode the output
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text

if __name__ == "__main__":
    model_name = "/Users/hanseung-yu/cpu-inference-server/models/llama3/llama3/Meta-Llama-3-8B"  # Correct path to your model
    input_text = "Once upon a time"
    try:
        output_text = test_llama_model(model_name, input_text)
        print("Generated text:")
        print(output_text)
    except Exception as e:
        print(f"An error occurred: {e}")

import sentencepiece as spm

def verify_tokenizer(model_file):
    sp = spm.SentencePieceProcessor()
    try:
        sp.load(model_file)
        print("Tokenizer model loaded successfully.")
        print(sp.encode_as_pieces("This is a test."))
    except Exception as e:
        print(f"Failed to load tokenizer model: {e}")

if __name__ == "__main__":
    model_file = './models/llama3/llama3/Meta-Llama-3-8B/tokenizer.model'
    verify_tokenizer(model_file)

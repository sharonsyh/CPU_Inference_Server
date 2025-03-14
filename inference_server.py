from flask import Flask, request, jsonify
from llama import Llama, Dialog

app = Flask(__name__)

# Load the model and tokenizer
generator = Llama.build(
    ckpt_dir="/Users/hanseung-yu/cpu-inference-server/models/llama3/Meta-Llama-3-8B",
    tokenizer_path="/Users/hanseung-yu/cpu-inference-server/models/llama3/Meta-Llama-3-8B/tokenizer.model",
    max_seq_len=16,
    max_batch_size=1,
)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    dialogs = data['dialogs']
    results = generator.chat_completion(
        dialogs,
        max_gen_len=None,
        temperature=0.6,
        top_p=0.9,
    )
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

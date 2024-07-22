from flask import Flask, request, jsonify, render_template
from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2LMHeadModel
import json
import torch

app = Flask(__name__)

# Set seed for reproducibility
set_seed(42)

# Load a larger transformer model for text generation
model_name = 'gpt2-medium'  # Change the model to gpt2-medium
generator = pipeline('text-generation', model=model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load the combined corpus
try:
    with open('combined_corpus.json', 'r') as file:
        combined_data = json.load(file)
    print("Combined corpus loaded successfully.")
except FileNotFoundError:
    print("Error: 'combined_corpus.json' file not found.")
    combined_data = {'information': '', 'qa': []}
except json.JSONDecodeError:
    print("Error: 'combined_corpus.json' is not a valid JSON file.")
    combined_data = {'information': '', 'qa': []}

def generate_response(user_input):
    try:
        print(f"Generating response for user input: {user_input}")
        context = combined_data['information'] + '\n\n' + json.dumps(combined_data['qa'], indent=2)
        
        context_tokens = tokenizer(context, return_tensors='pt')['input_ids'][0]
        user_input_tokens = tokenizer(user_input, return_tensors='pt')['input_ids'][0]
        bot_prefix_tokens = tokenizer("Bot:", return_tensors='pt')['input_ids'][0]

        total_length = len(context_tokens) + len(user_input_tokens) + len(bot_prefix_tokens) + 3
        if total_length > 1024:
            max_context_length = 1024 - (len(user_input_tokens) + len(bot_prefix_tokens) + 3)
            context_tokens = context_tokens[-max_context_length:]

        prompt_tokens = torch.cat([context_tokens, user_input_tokens, bot_prefix_tokens], dim=0).unsqueeze(0)
        prompt = tokenizer.decode(prompt_tokens[0], skip_special_tokens=True)
        print(f"Truncated Prompt: {prompt}")

        response = generator(prompt, max_new_tokens=100, pad_token_id=50256)
        
        if not response or not isinstance(response, list) or len(response) == 0 or 'generated_text' not in response[0]:
            raise ValueError("Invalid response structure from the generator.")
        
        generated_text = response[0]['generated_text']
        print(f"Generated text: {generated_text}")

        if "Bot:" in generated_text:
            bot_response = generated_text.split('Bot:')[-1].strip()
        else:
            bot_response = generated_text.strip()

        print(f"Generated response: {bot_response}")
        return bot_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Sorry, I encountered an error while generating the response: {e}"

@app.route('/')
def index():
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['message']
        print(f"Received message: {user_input}")
        response = generate_response(user_input)
        print(f"Sending response: {response}")
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in /chat route: {e}")
        return jsonify({'response': f"Sorry, there was an error processing your request: {e}"})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)

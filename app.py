from flask import Flask, render_template, request, jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM

import transformers

import torch


app = Flask(__name__)

model_id = "cognitivecomputations/dolphin-vision-72b" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


@app.route('/chatbot',
methods = ['POST'])

    

def chatbot():
    user_input = request.json['user_input']

    input_ids = tokenizer.encode(user_input, return_tensors ='pt')

    output_ids = model.generate(input_ids, max_length=200, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=1)[0]

    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    return jsonify({'response':response})

    

if __name__=='__main__':
    app.run(debug=True)
    

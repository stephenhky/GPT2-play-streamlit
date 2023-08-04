
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = st.radio('gpt2', ['gpt2', 'gpt2-large'])

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

input_text = st.text_input('Input text: ')

input_ids = tokenizer.encode(input_text, return_tensors='pt')

max_length = st.number_input('max_length', min_value=1, max_value=None, value=10000)
num_beams = st.number_input('num_beams', min_value=1, max_value=10, value=5)
no_repeat_ngram_size = st.number_input('no_repeat_ngram_size', min_value=0, value=2)
early_stopping = st.checkbox('early_stopping', value=True)

if st.button('Generate!'):
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping
    )

    generated_text = tokenizer.decode(outputs[0], skip_tokens=True)

    st.write(generated_text)

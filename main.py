
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = st.radio('gpt2', ['gpt2', 'gpt2-medium', 'gpt2-large'])

input_text = st.text_area('Input text: ')

max_length = st.number_input('max_length', min_value=1, max_value=None, value=10000)
num_beams = st.number_input('num_beams', min_value=1, max_value=10, value=5)
no_repeat_ngram_size = st.number_input('no_repeat_ngram_size', min_value=0, value=2)
do_sample = st.checkbox('do_sample', False)
early_stopping = st.checkbox('early_stopping', value=True)
temperature = st.number_input('temperature', min_value=0.0, max_value=10.0, value=1.0)
top_k = st.number_input('top_k', min_value=0, value=50)
top_p = st.number_input('top_p', min_value=0.0, max_value=1.0, value=1.0)


if st.button('Generate!'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample,
        early_stopping=early_stopping,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    generated_text = tokenizer.decode(outputs[0], skip_tokens=True)

    st.write(generated_text)

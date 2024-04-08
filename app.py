import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_community.llms import HuggingFaceHub
import os


def getLlamaResponse(input_text, num_words, blog_style):
    llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    # llm = HuggingFaceHub(repo_id="TheBloke/Llama-2-7B-Chat-GGML",
    #                      model_kwargs={"temperature": 0.1, "max_length": 64})
    template = """
    Write a blog for {blog_style} job profile for a 
    topic {input_text} within {num_word} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "num_word"],
                            template=template)

    response = llm(prompt.format(blog_style=blog_style,
                                 input_text=input_text, num_word=num_words))
    # prompt_string = prompt.format(
    #     blog_style=blog_style, input_text=input_text, num_word=num_words)

    # prompts = [prompt_string]

    # response = llm.generate(prompts)
    print(response)
    return response


st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    num_words = st.text_input('Number of Words')
with col2:
    blog_style = st.selectbox(
        'Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

if submit:
    st.write(getLlamaResponse(input_text, num_words, blog_style))

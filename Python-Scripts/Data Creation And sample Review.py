#!/usr/bin/env python
# coding: utf-8

# In[240]:


import pdfplumber

pdf_path1 = 'To Kill A Mockingbird - Full Text PDF.pdf'
pdf_path2 = "Frank Herbert's - Dune - Part 1 [EnglishOnlineClub.com].pdf"

def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text


# In[241]:


text1 = extract_text_from_pdf(pdf_path1)
text2 = extract_text_from_pdf(pdf_path2)


# In[242]:


import re

def clean_text(input_text):
    input_text = input_text.lower()
    # Remove newlines
    text_no_newlines = re.sub(r'\n', ' ', input_text)

    # Replace multiple spaces with a single space
    text_no_extra_spaces = re.sub(r'\s+', ' ', text_no_newlines)

    # Remove characters other than numbers and alphabets
    cleaned_text = re.sub(r'[^a-zA-Z0-9 ]', '', text_no_extra_spaces)

    return cleaned_text


# In[265]:


from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm  

def prepare_dataset(input_text, max_words_per_row=100):
    input_text = input_text.split()
    ret_data = []
    for i in range(0,len(input_text),max_words_per_row):
        text = ' '.join(input_text[i:i+max_words_per_row])
        ret_data.append(text + ',')
    
    return ret_data


# In[266]:


clean_text1 = clean_text(text1)
clean_text2 = clean_text(text2)


dataset1 = prepare_dataset(clean_text1)
dataset2 = prepare_dataset(clean_text2)


# In[275]:


len(dataset1),len(dataset2)


# In[276]:


dataset1[0]


# In[277]:


for i, inp in enumerate(dataset1[:5]):
    print(f"Example {i+1}:")
    print("Input:", inp)
    print()


# In[278]:


for i, input_segment, in enumerate(dataset2[:5]):
    print(f"Example {i+1}:")
    print("Input:", input_segment)
    print()


# In[279]:


# combine two Datasets
dataset = dataset1 + dataset2


# In[288]:


# for training
with open('train.txt','w') as f:
    for i in dataset[:len(dataset)-100]:
        f.write(i)

with open('test.txt','w') as f:
    for i in dataset[len(dataset)-100:]:
        f.write(i)


# In[289]:


from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_dir = r"C:\Users\aredd\Desktop\EndToEnd\text-completion\output_directory"


model = GPT2LMHeadModel.from_pretrained(model_dir, ignore_mismatched_sizes=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)


# In[295]:


prompt = """
Once upon a time there lived a 
"""

input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=150, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)


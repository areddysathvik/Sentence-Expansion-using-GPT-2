#!/usr/bin/env python
# coding: utf-8

# In[1]:


from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)


model_args = LanguageModelingArgs()
model_args.output_dir = "output_directory"
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 10
model_args.dataset_type = "simple"
model_args.mlm = False
model_args.max_length = 1024
model_args.use_cuda = True


# In[2]:


model = LanguageModelingModel(
    "gpt2",
    "gpt2-medium",
    args=model_args,
)


# In[3]:


train_path = "/content/drive/MyDrive/train.txt"
test_path = "/content/drive/MyDrive/test.txt"


# In[4]:


with open(train_path) as f:
  train_txt = f.read()

with open(test_path) as f:
  test_txt = f.read()


# In[5]:


train_data = train_txt.split(',')
test_data = test_txt.split(',')


# In[6]:


from tempfile import NamedTemporaryFile
import os


with NamedTemporaryFile(mode="w", delete=False) as train_file:
    for example in train_data:
        train_file.write(example + "\n")

with NamedTemporaryFile(mode="w", delete=False) as eval_file:
    for example in test_data:
        eval_file.write(example + "\n")


train_loss = model.train_model(train_file.name, eval_file=eval_file.name)

# Clean up temporary files
os.remove(train_file.name)
os.remove(eval_file.name)


# In[7]:


from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_dir = "/content/output_directory"


model = GPT2LMHeadModel.from_pretrained(model_dir, ignore_mismatched_sizes=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)


# In[9]:


prompt = """
Once upon a time
"""

input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=150, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)


# In[15]:


import shutil

folder_path = "/content/output_directory"
output_path = "/content/output_directory.zip"

shutil.make_archive(output_path.split('.')[0], 'zip', folder_path)


from nltk.tokenize import RegexpTokenizer
import numpy as np

path = 'data.txt'
text = open(path, "r", encoding='utf-8').read().lower()
# Tokineser ordende
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(text)
# fjerner gentagende tokens
unique_tokens = np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
text = "he"
if text in unique_token_index:
    print("True")
else:
    print("False")
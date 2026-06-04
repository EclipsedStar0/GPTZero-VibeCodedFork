"""
This code a slight modification of perplexity by hugging face
https://huggingface.co/docs/transformers/perplexity

Both this code and the orignal code are published under the MIT license.

by Burhan Ul tayyab and Nicholas Chua
"""

from model import GPT2PPL
from typing import List

# initialize the model
model = GPT2PPL()

print("Please enter your sentence: (Press Enter twice to start processing)")
contents: List[str] = []
while True:
    line: str = input()
    if len(line) == 0:
        break
    contents.append(line)
sentence: str = "\n".join(contents)

model(sentence)
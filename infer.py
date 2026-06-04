"""
This code a slight modification of perplexity by hugging face
https://huggingface.co/docs/transformers/perplexity

Both this code and the orignal code are published under the MIT license.

by Burhan Ul tayyab and Nicholas Chua
"""

from model import GPT2PPL
from typing import Tuple, Dict, Union

# initialize the model
model: GPT2PPL = GPT2PPL()

sentence: str = "your text here"

result: Tuple[Dict[str, Union[float, int]], str] = model(sentence)
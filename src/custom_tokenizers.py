import re
from functools import reduce

word_regex = r"(?:[^.,!?;¿\s]|\?\?\?)+" # Matches any string not containing punctuation or whitespace
def morpheme_tokenize_no_punc(str: str):
    """Tokenizes by splitting into morphemes, skipping punctuation"""
    words = re.findall(word_regex, str)
    words = [word.split('-') for word in words]
    words = [[morpheme for morpheme in word if morpheme != ''] for word in words]  # Remove empty morphemes introduced by faulty segmentation
    words = [word for word in words if word != []]
    morphemes = reduce(lambda a,b: a + ['[SEP]'] + b, words)
    return morphemes
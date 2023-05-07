"""Defines a tokenizer that uses multiple distinct vocabularies"""
from typing import List
import torch
import pickle
import random

special_chars = ["[UNK]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"]


def create_vocab(sentences: List[List[str]], threshold=2, should_not_lower=False):
    """Creates a set of the unique words in a list of sentences, only including words that exceed the threshold"""
    all_words = dict()
    for sentence in sentences:
        if sentence is None:
            continue
        for word in sentence:
            # Grams should stay uppercase, stems should be lowered
            if not word.isupper() and not should_not_lower:
                word = word.lower()
            if word not in special_chars:
                all_words[word] = all_words.get(word, 0) + 1

    all_words_list = []
    for word, count in all_words.items():
        if count >= threshold:
            all_words_list.append(word)

    return sorted(all_words_list)


class CustomEncoder:
    """Encodes and decodes words to an integer representation"""

    def __init__(self, vocabulary: List[str], output_vocabulary: List[str]=None):
        """
        :param vocabularies: A list of vocabularies for the tokenizer
        """
        self.vocabulary = vocabulary
        self.output_vocabulary = output_vocabulary
        self.special_chars = special_chars
        self.all_input_vocab = special_chars + vocabulary

        self.PAD_ID = special_chars.index("[PAD]")
        self.SEP_ID = special_chars.index("[SEP]")
        self.MASK_ID = special_chars.index("[MASK]")

    def encode_word(self, word: str, vocab: str) -> int:
        """Converts a word to the integer encoding
        :param word: The word to encode
        :param vocab: 'input' or 'output
        :return: An integer encoding
        """
        if vocab == 'input':
            if word in self.special_chars:
                return special_chars.index(word)
            elif word in self.vocabulary:
                return self.vocabulary.index(word)
            return 0
        elif vocab == 'output':
            if word in self.output_vocabulary:
                return self.output_vocabulary.index(word)
            else:
                print(word)
        else:
            raise ValueError("`vocab` must be either 'input' or 'output'")

    def encode(self, sentence: List[str], vocab: str) -> List[int]:
        """Encodes a sentence (a list of strings)
        :param sentence: The sentence to encode, a list of strings
        :param vocab: Should be 'input' or 'output'
        """
        return [self.encode_word(word, vocab=vocab) for word in sentence]

    def batch_decode(self, batch, vocab: str):
        """Decodes a batch of indices to the actual words
        :param batch: The batch of ids
        :param vocab: Should be 'input' or 'output'
        """
        def decode(seq):
            if isinstance(seq, torch.Tensor):
                indices = seq.detach().cpu().tolist()
            else:
                indices = seq.tolist()

            if vocab == 'input':
                return ['[UNK]' if index == 0 else self.all_input_vocab[index] for index in indices if (index >= len(special_chars) or index == 0)]
            elif vocab == 'output':
                return [self.output_vocabulary[index] for index in indices if index < len(self.output_vocabulary) and index >= 0]

        return [decode(seq) for seq in batch]

    def vocab_size(self):
        return len(self.all_input_vocab)

    def save(self):
        """Saves the encoder to a file"""
        with open('encoder_data.pkl', 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    def random_token_id(self):
        return random.randint(len(self.special_chars), len(self.special_chars) + len(self.vocabulary) - 1)


def load_encoder(path) -> CustomEncoder:
    with open(path, 'rb') as inp:
        return pickle.load(inp)
"""Defines a tokenizer that uses multiple distinct vocabularies"""
from typing import List
import torch
import pickle

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

    def __init__(self, vocabulary: List[str]):
        """
        :param vocabularies: A list of vocabularies for the tokenizer
        """
        self.vocabulary = vocabulary
        self.special_chars = special_chars
        self.all_vocab = special_chars + vocabulary

        self.PAD_ID = special_chars.index("[PAD]")
        self.SEP_ID = special_chars.index("[SEP]")
        self.BOS_ID = special_chars.index("[BOS]")
        self.EOS_ID = special_chars.index("[EOS]")

    def encode_word(self, word: str) -> int:
        """Converts a word to the integer encoding
        :param word: The word to encode
        :return: An integer encoding
        """

        if word in self.special_chars:
            return special_chars.index(word)
        elif word in self.vocabulary:
            return self.vocabulary.index(word)
        else:
            return 0

    def encode(self, sentence: List[str], vocabulary_index, separate_vocab=False) -> List[int]:
        """Encodes a sentence (a list of strings)"""
        return [self.encode_word(word) for word in sentence]

    def batch_decode(self, batch):
        """Decodes a batch of indices to the actual words
        :param batch: The batch of ids
        """
        def decode(seq):
            if isinstance(seq, torch.Tensor):
                indices = seq.detach().cpu().tolist()
            else:
                indices = seq.tolist()
            return ['[UNK]' if index == 0 else self.all_vocab[index] for index in indices if (index >= len(special_chars) or index == 0)]

        return [decode(seq) for seq in batch]

    def vocab_size(self):
        return len(self.all_vocab)

    def save(self):
        """Saves the encoder to a file"""
        with open('encoder_data.pkl', 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)


def load_encoder(path) -> CustomEncoder:
    with open(path, 'rb') as inp:
        return pickle.load(inp)
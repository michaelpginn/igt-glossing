import re
from abc import ABC
from functools import reduce
from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional, Tuple, Any

word_regex = r"(?:[^.,!?;¿\s]|\?\?\?)+" # Matches any string not containing punctuation or whitespace
def morpheme_tokenize_no_punc(str: str):
    """Tokenizes by splitting into morphemes, skipping punctuation"""
    words = re.findall(word_regex, str)
    words = [word.split('-') for word in words]
    words = [[morpheme for morpheme in word if morpheme != ''] for word in words]  # Remove empty morphemes introduced by faulty segmentation
    words = [word for word in words if word != []]
    morphemes = reduce(lambda a,b: a + ['[SEP]'] + b, words)
    return morphemes


class WordLevelTokenizer(PreTrainedTokenizer):
    """
    Constructs a tokenizer for Roberta architecture models that just encodes using a simple word-level integer encoding
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab: List[str], model_max_length: int):
        self.special_chars = ["[UNK]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab = vocab
        self.all_vocab = self.special_chars + self.vocab

        self.UNK_ID = self.special_chars.index("[UNK]")
        self.SEP_ID = self.special_chars.index("[SEP]")
        self.PAD_ID = self.special_chars.index("[PAD]")
        self.MASK_ID = self.special_chars.index("[MASK]")

        super().__init__(
            errors="replace",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            mask_token="[MASK]",
            add_prefix_space=False
        )

        self.model_max_length = model_max_length

    @property
    def vocab_size(self) -> int:
        return len(self.all_vocab)

    def __len__(self):
        return len(self.all_vocab)

    def get_vocab(self) -> Dict[str, int]:
        return {word: index for word, index in zip(self.all_vocab, range(len(self.all_vocab)))}

    def _tokenize(self, text, **kwargs):
        raise NotImplementedError

    def _convert_token_to_id(self, token):
        return self.all_vocab.index(token) if token in self.all_vocab else self.UNK_ID

    def _convert_id_to_token(self, index: int) -> str:
        return self.all_vocab[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        raise NotImplementedError

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        raise NotImplementedError

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if token_ids_1 is not None:
            raise NotImplementedError
        return [1 if token == self.PAD_ID else 0 for token in token_ids_0]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        raise NotImplementedError

    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def tokenize(self, text: str, **kwargs) -> List[str]:
        raise NotImplementedError

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        raise NotImplementedError
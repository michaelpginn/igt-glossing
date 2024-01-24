"""Defines models and functions for loading, manipulating, and writing task data"""
import os.path
import re
from functools import reduce
from typing import Optional, List, Union

import pandas as pd
import torch
from datasets import Dataset, DatasetDict

from tokenizer import morpheme_tokenize_no_punc as tokenizer, WordLevelTokenizer
from transformers import Trainer, PreTrainedTokenizerFast, PreTrainedTokenizer

special_chars = ["[UNK]", "[SEP]", "[PAD]", "[MASK]"]


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


def create_gloss_vocab(morphology):
    def parse_tree(morphology_subtree):
        all_glosses = []
        for item in morphology_subtree:
            if isinstance(item, tuple):
                all_glosses += parse_tree(item[1])
            else:
                all_glosses.append(item)
        return all_glosses

    return parse_tree(morphology)


# def prepare_dataset_mlm(data: List[List[str]], tokenizer: WordLevelTokenizer, device):
#     """Encodes, pads, and creates attention mask for input."""
#
#     # Create a dataset
#     raw_dataset = Dataset.from_list([{'tokens': line} for line in data])
#
#     def process(row):
#         source_enc = tokenizer.convert_tokens_to_ids(row['tokens'])
#
#         # Encode the output, if present
#         return {'input_ids': torch.tensor(source_enc, dtype=torch.long).to(device)}
#
#     return raw_dataset.map(process)


def prepare_dataset(dataset: DatasetDict, train_vocab, tokenizer, glosses: list[str]):
    """Encodes and pads inputs and creates attention mask"""

    def tokenize_and_align_labels(row):
        morphemes = [[morph if morph in train_vocab else tokenizer.unk_token for morph in morphemes] for morphemes in
                     row["morphemes"]]
        tokenized_inputs = tokenizer(morphemes, truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(row["glosses"]):
            if isinstance(tokenizer, PreTrainedTokenizerFast):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        gloss = label[word_idx]
                        label_ids.append(glosses.index(gloss) if gloss in glosses else glosses.index('<unk>'))
                    previous_word_idx = word_idx
            else:
                # Slow tokenizer
                label_ids = [glosses.index(gloss) if gloss in glosses else glosses.index('<unk>') for gloss in label]
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False, batch_size=1)


# def prepare_dataset(data: List[IGTLine], tokenizer: WordLevelTokenizer, labels: list[str], device):
#     """Encodes and pads inputs and creates attention mask"""
#
#     # Create a dataset
#     raw_dataset = Dataset.from_list([line.__dict__() for line in data])
#
#     def process(row):
#         source_enc = tokenizer.convert_tokens_to_ids(row['morphemes'])
#
#         # Pad
#         initial_length = len(source_enc)
#         source_enc += [tokenizer.PAD_ID] * (tokenizer.model_max_length - initial_length)
#
#         # Create attention mask
#         attention_mask = [1] * initial_length + [0] * (tokenizer.model_max_length - initial_length)
#
#         # Encode the output, if present
#         if 'glosses' in row:
#             # For token class., the labels are just the glosses for each word
#             output_enc = [labels.index(gloss) for gloss in row['glosses']]
#             output_enc += [-100] * (tokenizer.model_max_length - len(output_enc))
#             return {'input_ids': torch.tensor(source_enc, dtype=torch.long).to(device),
#                     'attention_mask': torch.tensor(attention_mask).to(device),
#                     'labels': torch.tensor(output_enc, dtype=torch.long).to(device)}
#
#         else:
#             # If we have no glosses, this must be a prediction task
#             return {'input_ids': torch.tensor(source_enc).to(device),
#                     'attention_mask': torch.tensor(attention_mask).to(device)}
#
#     return raw_dataset.map(process)


def split_line(line: str, prefix: Union[str, None]):
    words = line.split()
    # Insert [SEP] between words
    words = reduce(lambda r, v: r + ["<sep>", v], words[1:], words[:1])
    morphemes = [word.split('-') for word in words]
    return [prefix + morpheme if morpheme != '<sep>' and prefix is not None else morpheme
            for word in morphemes for morpheme in word if morpheme != '']

# def prepare_multitask_dataset(data: List[IGTLine], tokenizer: WordLevelTokenizer, labels: list[str], device):
#     """Encodes and pads inputs and creates attention mask"""
#
#     # Create a dataset
#     raw_dataset = Dataset.from_list([line.__dict__() for line in data])
#
#     morphology = pd.read_csv('./uspanteko_morphology.csv')
#
#     def process(row):
#         source_enc = tokenizer.convert_tokens_to_ids(row['morphemes'])
#
#         # Pad
#         initial_length = len(source_enc)
#         source_enc += [tokenizer.PAD_ID] * (tokenizer.model_max_length - initial_length)
#
#         # Create attention mask
#         attention_mask = [1] * initial_length + [0] * (tokenizer.model_max_length - initial_length)
#
#         # Encode the output, if present
#         if 'glosses' in row:
#             all_labels_enc = []
#
#             # Create labels for every level of hierarchy in the morphology
#             for level in range(morphology.shape[1]):
#                 if level == 0:
#                     output_enc = [labels.index(gloss) for gloss in row['glosses']]
#                 else:
#                     output_enc = [morphology[morphology['Gloss'] == gloss].iloc[0, level] for gloss in row['glosses']]
#                 output_enc += [-100] * (tokenizer.model_max_length - len(output_enc))
#                 all_labels_enc.append(torch.tensor(output_enc, dtype=torch.long))
#
#             return {'input_ids': torch.tensor(source_enc, dtype=torch.long).to(device),
#                     'attention_mask': torch.tensor(attention_mask).to(device),
#                     'labels': torch.stack(all_labels_enc).to(device)}
#
#         else:
#             # If we have no glosses, this must be a prediction task
#             return {'input_ids': torch.tensor(source_enc).to(device),
#                     'attention_mask': torch.tensor(attention_mask).to(device)}
#
#     return raw_dataset.map(process)


# def write_predictions(data: List[IGTLine], tokenizer, trainer: Trainer, labels, out_path):
#     """Runs predictions for a dataset and writes the output IGT"""
#     dataset = prepare_dataset(data=data, tokenizer=tokenizer, labels=labels, device='cpu')
#     preds = trainer.predict(dataset).predictions
#     decoded_preds = [[labels[index] for index in pred_seq if len(labels) > index >= 0] for pred_seq in preds]
#
#     with open(out_path, 'w') as outfile:
#         for line, line_preds in zip(data, decoded_preds):
#             # Write the data in the appropriate format
#             outfile.write("\\t " + line.transcription)
#             outfile.write("\n\\m " + line.segmentation)
#
#             # Trim preds to the number of morphemes
#             line_preds = line_preds[:len(line.morphemes())]
#             # Combine predictions into a string
#             line_pred_string = "\n\\p "
#             for pred_gloss in line_preds:
#                 if pred_gloss == "[SEP]":
#                     line_pred_string += " "
#                 elif line_pred_string[-1] == " ":
#                     line_pred_string += pred_gloss
#                 else:
#                     line_pred_string += "-" + pred_gloss
#
#             outfile.write(line_pred_string)
#             outfile.write("\n\\l " + line.translation + "\n\n")


# def write_igt(data: List[IGTLine], out_path):
#     with open(out_path, 'w') as outfile:
#         for line in data:
#             # Write the data in the appropriate format
#             outfile.write("\\t " + line.transcription)
#             outfile.write("\n\\m " + line.segmentation)
#             outfile.write("\n\\p " + line.glosses)
#             outfile.write("\n\\l " + line.translation + "\n\n")

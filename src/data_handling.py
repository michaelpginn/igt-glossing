"""Defines models and functions for loading, manipulating, and writing task data"""
import os.path
import re
from functools import reduce
from typing import Optional, List

import pandas as pd
import torch
from datasets import Dataset

from tokenizer import morpheme_tokenize_no_punc as tokenizer, WordLevelTokenizer
from transformers import Trainer

special_chars = ["[UNK]", "[SEP]", "[PAD]", "[MASK]"]


class IGTLine:
    """A single line of IGT"""

    def __init__(self, transcription: str, segmentation: Optional[str], glosses: Optional[str],
                 translation: Optional[str]):
        self.transcription = transcription
        self.segmentation = segmentation
        self.glosses = glosses
        self.translation = translation
        self.should_segment = True

    def __repr__(self):
        return f"Trnsc:\t{self.transcription}\nSegm:\t{self.segmentation}\nGloss:\t{self.glosses}\nTrnsl:\t{self.translation}\n\n"

    def gloss_list(self, segmented=False) -> Optional[List[str]]:
        """Returns the gloss line of the IGT as a list.
        :param segmented: If True, will return each morpheme gloss as a separate item.
        """
        if self.glosses is None:
            return []
        if not segmented:
            return self.glosses.split()
        else:
            words = re.split("\s+", self.glosses)
            glosses = [re.split("-", word) for word in words]
            glosses = [[gloss.replace('.', '') for gloss in word_glosses if gloss != ''] for word_glosses in
                       glosses]  # Remove empty glosses introduced by faulty segmentation
            glosses = [word_glosses for word_glosses in glosses if word_glosses != []]
            glosses = reduce(lambda a, b: a + ['[SEP]'] + b, glosses)  # Add separator for word boundaries
            return glosses

    def morphemes(self) -> Optional[List[str]]:
        """Returns the segmented list of morphemes, if possible"""
        if self.segmentation is None:
            return None
        return tokenizer(self.segmentation)

    def __dict__(self):
        d = {'transcription': self.transcription, 'translation': self.translation}
        if self.glosses is not None:
            d['glosses'] = self.gloss_list(segmented=self.should_segment)
        if self.segmentation is not None:
            d['segmentation'] = self.segmentation
            d['morphemes'] = self.morphemes()
        return d


def load_data_file(path: str) -> List[IGTLine]:
    """Loads a file containing IGT data into a list of entries."""
    all_data = []

    # If we have a directory, recursively load all files and concat together
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(".txt"):
                all_data.extend(load_data_file(os.path.join(path, file)))
        return all_data

    # If we have one file, read in line by line
    with open(path, 'r') as file:
        current_entry = [None, None, None, None]  # transc, segm, gloss, transl

        for line in file:
            # Determine the type of line
            # If we see a type that has already been filled for the current entry, something is wrong
            line_prefix = line[:2]
            if line_prefix == '\\t' and current_entry[0] == None:
                current_entry[0] = line[3:].strip()
            elif line_prefix == '\\m' and current_entry[1] == None:
                current_entry[1] = line[3:].strip()
            elif line_prefix == '\\p' and current_entry[2] == None:
                if len(line[3:].strip()) > 0:
                    current_entry[2] = line[3:].strip()
            elif line_prefix == '\\l' and current_entry[3] == None:
                current_entry[3] = line[3:].strip()
                # Once we have the translation, we've reached the end and can save this entry
                all_data.append(IGTLine(transcription=current_entry[0],
                                        segmentation=current_entry[1],
                                        glosses=current_entry[2],
                                        translation=current_entry[3]))
                current_entry = [None, None, None, None]
            elif line.strip() != "":
                # Something went wrong
                continue
            else:
                if not current_entry == [None, None, None, None]:
                    all_data.append(IGTLine(transcription=current_entry[0],
                                            segmentation=current_entry[1],
                                            glosses=current_entry[2],
                                            translation=None))
                    current_entry = [None, None, None, None]
        # Might have one extra line at the end
        if not current_entry == [None, None, None, None]:
            all_data.append(IGTLine(transcription=current_entry[0],
                                    segmentation=current_entry[1],
                                    glosses=current_entry[2],
                                    translation=None))
    return all_data


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


def prepare_dataset_mlm(data: List[List[str]], tokenizer: WordLevelTokenizer, device):
    """Encodes, pads, and creates attention mask for input."""

    # Create a dataset
    raw_dataset = Dataset.from_list([{'tokens': line} for line in data])

    def process(row):
        source_enc = tokenizer.convert_tokens_to_ids(row['tokens'])

        # Encode the output, if present
        return {'input_ids': torch.tensor(source_enc, dtype=torch.long).to(device)}

    return raw_dataset.map(process)


def prepare_dataset(data: List[IGTLine], tokenizer: WordLevelTokenizer, labels: list[str], device):
    """Encodes and pads inputs and creates attention mask"""

    # Create a dataset
    raw_dataset = Dataset.from_list([line.__dict__() for line in data])

    def process(row):
        source_enc = tokenizer.convert_tokens_to_ids(row['morphemes'])

        # Pad
        initial_length = len(source_enc)
        source_enc += [tokenizer.PAD_ID] * (tokenizer.model_max_length - initial_length)

        # Create attention mask
        attention_mask = [1] * initial_length + [0] * (tokenizer.model_max_length - initial_length)

        # Encode the output, if present
        if 'glosses' in row:
            # For token class., the labels are just the glosses for each word
            output_enc = [labels.index(gloss) for gloss in row['glosses']]
            output_enc += [-100] * (tokenizer.model_max_length - len(output_enc))
            return {'input_ids': torch.tensor(source_enc, dtype=torch.long).to(device),
                    'attention_mask': torch.tensor(attention_mask).to(device),
                    'labels': torch.tensor(output_enc, dtype=torch.long).to(device)}

        else:
            # If we have no glosses, this must be a prediction task
            return {'input_ids': torch.tensor(source_enc).to(device),
                    'attention_mask': torch.tensor(attention_mask).to(device)}

    return raw_dataset.map(process)


def prepare_multitask_dataset(data: List[IGTLine], tokenizer: WordLevelTokenizer, labels: list[str], device):
    """Encodes and pads inputs and creates attention mask"""

    # Create a dataset
    raw_dataset = Dataset.from_list([line.__dict__() for line in data])

    morphology = pd.read_csv('./uspanteko_morphology.csv')

    def process(row):
        source_enc = tokenizer.convert_tokens_to_ids(row['morphemes'])

        # Pad
        initial_length = len(source_enc)
        source_enc += [tokenizer.PAD_ID] * (tokenizer.model_max_length - initial_length)

        # Create attention mask
        attention_mask = [1] * initial_length + [0] * (tokenizer.model_max_length - initial_length)

        # Encode the output, if present
        if 'glosses' in row:
            all_labels_enc = []

            # Create labels for every level of hierarchy in the morphology
            for level in range(morphology.shape[1]):
                if level == 0:
                    output_enc = [labels.index(gloss) for gloss in row['glosses']]
                else:
                    output_enc = [morphology[morphology['Gloss'] == gloss].iloc[0, level] for gloss in row['glosses']]
                output_enc += [-100] * (tokenizer.model_max_length - len(output_enc))
                all_labels_enc.append(torch.tensor(output_enc, dtype=torch.long))

            return {'input_ids': torch.tensor(source_enc, dtype=torch.long).to(device),
                    'attention_mask': torch.tensor(attention_mask).to(device),
                    'labels': torch.stack(all_labels_enc).to(device)}

        else:
            # If we have no glosses, this must be a prediction task
            return {'input_ids': torch.tensor(source_enc).to(device),
                    'attention_mask': torch.tensor(attention_mask).to(device)}

    return raw_dataset.map(process)


def write_predictions(data: List[IGTLine], tokenizer, trainer: Trainer, labels, out_path):
    """Runs predictions for a dataset and writes the output IGT"""
    dataset = prepare_dataset(data=data, tokenizer=tokenizer, labels=labels, device='cpu')
    preds = trainer.predict(dataset)
    decoded_preds = [[labels[index] for index in pred_seq if len(labels) > index >= 0] for pred_seq in preds]

    with open(out_path, 'w') as outfile:
        for line, line_preds in zip(data, decoded_preds):
            # Write the data in the appropriate format
            outfile.write("\\t " + line.transcription)
            outfile.write("\\m " + line.segmentation)

            # Trim preds to the number of morphemes
            line_preds = line_preds[:len(line.morphemes())]
            # Combine predictions into a string
            line_pred_string = "\\p "
            for pred_gloss in line_preds:
                if pred_gloss == "[SEP]":
                    line_pred_string += " "
                elif line_pred_string[-1] == " ":
                    line_pred_string += pred_gloss
                else:
                    line_pred_string += "-" + pred_gloss

            outfile.write(line_pred_string)
            outfile.write("\\l " + line.translation + "\n")

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import torch

from dictionary import Dictionary
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
from collections import Counter

from fairseq import tokenizer, utils
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line
# additional dictionay


class NormDictionary(Dictionary):
    """
    Adapted from (and inherits from) fairseq.data.dictionary

    Constructs normalised dictionary mapping for RRGen
    attribute values

    #NOTE currently not in use
    """

    def __init__(
        self,
        *,  # begin keyword-only arguments
        unk=0,
        extra_special_symbols=None,
    ):
        self.unk_word = unk
        self.symbols = []
        self.count = []
        self.indices = {}
        # self.bos_index = self.add_symbol(bos)
        # self.pad_index = self.add_symbol(pad)
        # self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        # assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        elif int(sym) in self.indices:
            return self.indices[int(sym)]
        else:
            return self.unk_index

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """

        scaler = MinMaxScaler()
        self.count = scaler.fit_transform(
            np.array(self.symbols).reshape(-1, 1)).flatten()

        self.indices = dict(zip(self.symbols, self.count))

    def read_files(self, filenames):
        data = Counter()
        for filename in filenames:
            with open(filename, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    try:
                        data[int(line)] += 1
                    except ValueError:
                        data[line] += 1

        data[self.unk_word] = 1

        self.symbols = sorted(list(data.keys()))

        return

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with PathManager.open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        # import pdb
        # pdb.set_trace()

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False

                # expects one-hot encoding, but ext_senti,
                # ext_cate, ext_rate are floats between [0, 1]
                count = float(field)

                word = int(line)
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file."
                        .format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    f"Incorrect dictionary format, expected '<int> <float>', got '{line} {field}'."
                )

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            PathManager.mkdirs(os.path.dirname(f))
            with PathManager.open(f, "w", encoding="utf-8") as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print("{} {}".format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        # ex_keys, ex_vals = self._get_meta()
        self._save(
            f,
            zip(
                self.symbols,
                self.count,
            ),
        )

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        return t

    def encode_line(
        self,
        line,
    ):

        idx = self.index(line.strip())

        return [idx]


if __name__ == "__main__":
    # for testing
    filename = '/srv/scratch2/kew/fairseq_materials/translation_format/train.senti'
    dict_file = '/srv/scratch2/kew/fairseq_materials/translation_format/test_senti_dict.txt'

    # d = NormDictionary()

    # # print(d)
    # d.read_files([filename])

    # d.finalize()
    # print(d.symbols)
    # print(d.count)
    # print(d.indices)

    # lines = ['0\n', '1\n', '2\n', '4\n', '5\n', '6\n', '7\n', '10\n']
    # for l in lines:
    #     print(d.encode_line(l))

    # save
    # d.save(dict_file)

    # load
    d = NormDictionary.load(dict_file)
    print(d.indices)

# -*- coding: utf-8 -*-

import enchant
from itertools import groupby

__all__ = ['load_dict', 'split']

_dictionary = None


def _is_in_dict(word):
    # Some languages (e.g. German) need some words capitalized
    return _dictionary.check(word.capitalize())


def load_dict(language='de_de'):
    global _dictionary
    _dictionary = enchant.Dict(language)


def split(word, language=None, min_word_size=3, split_on_s=True):
    if language is not None:
        load_dict(language)

    if _dictionary is None:
        load_dict()

    words = _split_crawler(word, min_word_size, split_on_s)

    # Groups words in sections such that each section contains only
    # words shorter than min word size or not shorter than min word
    # size.
    words_concat = []
    groups = [
        list(group[1])
        for group
        in groupby(
            words,
            lambda x: len(x) < min_word_size
        )
    ]

    # Concatenate sections containing shorter words than min word size.
    for group in groups:
        if len(group[0]) < min_word_size:
            group = [''.join(group)]
        words_concat.extend(group)

    # If the word is split in 2 and the 2nd word does not exist,
    # conclude it is not a compound and discard the split.
    word_is_not_compound = (
            len(words_concat) == 2
            and not _is_in_dict(words_concat[1])
    )
    if word_is_not_compound:
            return [''.join(words_concat)]

    return words_concat


def _split_crawler(word, min_word_size, split_on_s):
    word = word.lower()
    words = []

    for right_word_len in range(1, len(word)):
        split_idx = len(word) - right_word_len
        left_word = word[:split_idx]
        right_word = word[split_idx:]

        if not _is_in_dict(left_word):
            continue

        # Should help with letters 's' and alike in some languages
        if split_on_s:
            is_s = left_word[-1] == 's'
            has_length = len(left_word[:-1]) > min_word_size
            if is_s and has_length and _is_in_dict(left_word[:-1]):
                right_word = left_word[-1] + right_word
                left_word = left_word[:-1]

        words.append(left_word)

        if not _is_in_dict(right_word):
            words.extend(
                _split_crawler(right_word, min_word_size, split_on_s)
            )
            return words

        words.append(right_word)

        return words

    return [word]

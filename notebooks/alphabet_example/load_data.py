import numpy as np
from lang_utils import load_paths

N = 2000  # num words per language
langs = ["en", "de", "fr", "it", "pl"]  # which languages to use
path = "./dicts/"  # path to dictionaries
max_word_len = 20  # max length of path (will truncate words longer than this)
alpha_len = 26  # length of alphabet

words, paths, labels = load_paths(
    N, langs=langs, path=path, max_word_len=max_word_len, alpha_len=alpha_len
)

# start all paths with the empty word (all zeros)
empty_words = np.zeros((paths.shape[0], 1, paths.shape[2]))
paths = np.concatenate([empty_words, paths], axis=1)

import numpy as np
from numba import njit
from tqdm.auto import tqdm as tq


@njit()
def construct_path(char_seq, projection_matrix=np.identity(26), alpha_len=26):
    n = len(char_seq)
    its = np.zeros(n, np.int64)
    for i in range(n):
        its[i] = ord(char_seq[i]) - 97
    A = np.zeros((n + 1, alpha_len))
    j = 1
    for i in its:
        A[j, i] += 1
        j += 1

    A = A @ projection_matrix

    return A


def load_words(N, langs=["en"], path="data"):
    words_dict = {}
    for lang in langs:
        word_file = "%s/wordlist_%s.txt" % (path, lang)
        with open(word_file) as f:
            words = np.array(f.read().splitlines())
            # print(len(words))
        if N is not None:
            np.random.seed(1)
            words_dict[lang] = np.random.choice(words, size=N, replace=False)

    return words_dict, langs


def get_paths_from_words(words_dict, langs, max_word_len=20, alpha_len=26):
    words = np.hstack([words_dict[lang] for lang in langs])
    E_dict = {}
    for lang in langs:
        E_dict[lang] = np.array(
            [
                np.vstack(
                    [
                        np.zeros((50 - len(word), 26)),
                        construct_path(word),
                    ]
                )
                for word in tq(words_dict[lang])
            ]
        )
    E = np.vstack([E_dict[lang] for lang in langs])
    y = np.hstack([i * np.ones(len(E_dict[lang])) for i, lang in enumerate(langs)])
    E = E[:, -max_word_len:, :alpha_len]

    return words, E, y


def load_paths(N, langs=["en"], path="data", max_word_len=20, alpha_len=26):
    words_dict, langs = load_words(N, langs, path=path)
    words, E, y = get_paths_from_words(words_dict, langs, max_word_len, alpha_len)
    return words, E, y

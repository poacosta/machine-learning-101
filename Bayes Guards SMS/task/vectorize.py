import numpy as np
import string


# This function creates an array of lines, where lines are transformed into lists of words
# without spaces and punctuation symbols. `np.char.lower()` returns an array with the elements
# converted to lowercase; `translate()` returns a copy of the string where characters have
# been mapped through the given translation table; `str.maketrans()` provides a translation table
# for `translate()`, in our case it specifies that punctuation symbols should be replaced with
# None. `np.char.split()` returns a list of the words for each element in the array.
def split_by_words(X):
    return np.char.split(np.char.translate(np.char.lower(X), str.maketrans('', '', string.punctuation)))


def vectorize(X):
    # get the number of input messages
    X_len = len(X)
    # get a vector of words out of each message
    X_split = split_by_words(X)

    # get a 1D array of unique words
    uniques = np.unique(np.concatenate(X_split))
    # create an index dictionary and fill it with unique words and their indices
    index_dict = {}
    for index, word in enumerate(uniques):
        index_dict[word] = index

    # create an array of zeros with dimensions corresponding
    # to input data size and index_dict length
    vectorization = np.zeros((X_len, len(index_dict)))
    # each i'th line of the array contains in the j'th position a number x
    # which shows how many times the i'th word was encountered in the j'th message
    for index, message in enumerate(X_split):
        unique, count = np.unique(message, return_counts=True)
        for i, word in enumerate(unique):
            vectorization[index, index_dict[word]] = count[i]

    return index_dict, vectorization

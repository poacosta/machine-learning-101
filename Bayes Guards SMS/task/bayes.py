from vectorize import *


class NaiveBayes:

    def __init__(self):
        self.likelihood = None
        self.dict_size = None
        self.unique_classes = None
        self.dictionary = None
        self.classes_words_count = None
        self.classes_prior = None

    def fit(self, X, y):
        self.unique_classes = np.unique(y)

        self.dictionary, X = vectorize(X)
        self.dict_size = len(self.dictionary)

        self.classes_prior = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.classes_words_count = np.zeros(len(self.unique_classes), dtype=np.float64)
        # Initially, the likelihood array must be filled not with zeros but with α=1
        self.likelihood = np.full((len(self.unique_classes), self.dict_size + 1), 1, dtype=np.float64)

        for i, clazz in enumerate(self.unique_classes):
            y_i_mask = y == clazz
            y_i_sum = np.sum(y_i_mask)
            self.classes_prior[i] = y_i_sum / len(y)
            self.classes_words_count[i] = np.sum(X[y_i_mask])
            self.likelihood[i, :-1] += np.sum(X[y_i_mask], 0)

            denominator = self.classes_words_count[i]
            self.likelihood[i] = self.likelihood[i] / denominator

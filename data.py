import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from preprocess_data import preprocess_tweet


class Data:

    def __init__(self, path_train, path_test):
        self.TRAIN_DF = pd.read_csv(path_train, sep=',')
        self.TEST_DF = pd.read_csv(path_test, sep=',')
        self.x = self.TRAIN_DF.text
        self.y = self.TRAIN_DF.label

    def k_fold_train_test_sets(self):
        x_train_sets = []
        x_test_sets = []
        y_train_sets = []
        y_test_sets = []
        kf = KFold(n_splits=10, random_state=50, shuffle=True)
        for train_index, test_index in kf.split(self.x):
            x_train_sets.append(self.x[train_index])
            x_test_sets.append(self.x[test_index])
            y_train_sets.append(self.y[train_index])
            y_test_sets.append(self.y[test_index])

        return x_train_sets, y_train_sets, x_test_sets, y_test_sets

    def preprocess_data(self):
        self.TRAIN_DF.text = self.TRAIN_DF.text.apply(preprocess_tweet)
        self.TEST_DF.text = self.TEST_DF.text.apply(preprocess_tweet)

    @staticmethod
    def train_test_vectors(train, test, param, vec="COUNT"):

        if vec == "COUNT":
            vectorizer = CountVectorizer(max_features=param[0], max_df=param[1])
        elif vec == "TFIDF":
            vectorizer = TfidfVectorizer(max_features=param[0], max_df=param[1])

        x_train = vectorizer.fit_transform(train)
        x_test = vectorizer.transform(test)

        return x_train, x_test






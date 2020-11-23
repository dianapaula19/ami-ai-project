from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from data import Data


n_estimators = [2000, 2250, 2500, 2750, 3000]
max_depth = [100, 200, 300, 400, 500]
C = [0.1, 0.01, 0.001, 0.0001]


data = Data('./data/train.csv', './data/test.csv')
data.preprocess_data()
x_train, x_test = Data.train_test_vectors(data.TRAIN_DF.text, data.TEST_DF.text, [3000, 0.4], "TFIDF")

evc = VotingClassifier(estimators=[
    ('mnb', MultinomialNB()),
    ('lr', LogisticRegression(max_iter=1000)),
    ('svm', LinearSVC()),
    ('rf', RandomForestClassifier(random_state=2))
], voting='hard')

param_grid = {
    'svm__C': C,
    'rf__max_depth': max_depth,
    'rf__n_estimators': n_estimators
}

grid_search = GridSearchCV(estimator=evc, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)

grid_search.fit(x_train, data.TRAIN_DF.label)

print(grid_search.best_params_)





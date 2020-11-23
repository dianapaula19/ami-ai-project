from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def evc_model(x_train, y_train):

    mnb = MultinomialNB()

    lr = LogisticRegression(max_iter=1000)

    rf = RandomForestClassifier(n_estimators=2500, max_depth=250, random_state=2)

    svm = LinearSVC(C=0.0001)

    evc = VotingClassifier(estimators=[('mnb', mnb), ('lr', lr), ('rf', rf), ('svm', svm)], voting='hard')

    evc.fit(x_train, y_train)

    return evc


def score_on_model(x_train, y_train, x_test, y_test, model):

    if model == "VotingClassifier":

        evc = evc_model(x_train, y_train)

        return evc.score(x_train, y_train), evc.score(x_test, y_test)

    if model == "LogisticRegression":

        lr = LogisticRegression(max_iter=1000)
        lr.fit(x_train, y_train)

        return lr.score(x_train, y_train), lr.score(x_test, y_test)


def prediction(x_train, y_train, x_test, model):

    if model == "VotingClassifier":

        evc = evc_model(x_train, y_train)

        return evc.predict(x_test)

    if model == "LogisticRegression":

        lr = LogisticRegression(max_iter=1000)
        lr.fit(x_train, y_train)

        return lr.predict(x_test)

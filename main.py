from data import Data
from model import score_on_model, prediction
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt


def k_fold_mean_score(x_train_sets, y_train_sets, x_test_sets, y_test_set, model):

    print("Model: " + model)
    final_confusion_matrix = [[0, 0], [0, 0]]
    scores = []
    for x_train, y_train, x_test, y_test in zip(x_train_sets, y_train_sets, x_test_sets, y_test_set):
        x_train, x_test = Data.train_test_vectors(x_train, x_test, [3000, 0.4], "TFIDF")
        y_pred = prediction(x_train, y_train, x_test, model)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        final_confusion_matrix[0][0] += tn
        final_confusion_matrix[0][1] += fp
        final_confusion_matrix[1][0] += fn
        final_confusion_matrix[1][1] += tp
        scores.append(score_on_model(x_train, y_train, x_test, y_test, model))

    avg = np.mean(scores, axis=0)

    print("Score for train: " + str(avg[0]))
    print("Score for test: " + str(avg[1]))
    print(final_confusion_matrix)

    plt.figure(figsize=(10, 7))
    plt.title(model)
    sn.heatmap(final_confusion_matrix, annot=True, fmt='g', linewidths=10, cmap="YlGnBu")
    plt.savefig('./plots/confusion_matrix_' + model)


def write_submission(y_pred, submission_file):

    rows = []
    index = 5001

    for y in y_pred:
        rows.append([index, y])
        index += 1

    with open(submission_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "label"])
        writer.writerows(rows)


def main():

    data = Data('./data/train.csv', './data/test.csv')
    data.preprocess_data()
    x_train_sets, y_train_sets, x_test_sets, y_test_set = data.k_fold_train_test_sets()

    models = ["LogisticRegression", "VotingClassifier"]

    for model in models:

        k_fold_mean_score(x_train_sets, y_train_sets, x_test_sets, y_test_set, model)

    x_train, x_test = Data.train_test_vectors(data.TRAIN_DF.text, data.TEST_DF.text, [3000, 0.4], "TFIDF")
    y_train = data.TRAIN_DF.label

    print("Predictions for the LogisticRegression Model")
    y_pred = prediction(x_train, y_train, x_test, "LogisticRegression")

    write_submission(y_pred, "./submissions/lr_submission.csv")

    print("Predictions for the VotingClassifier Model")
    y_pred = prediction(x_train, y_train, x_test, "VotingClassifier")

    write_submission(y_pred, "./submissions/evc_submission.csv")


if __name__ == '__main__':
    main()

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing, model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns


def encode(train, test):
    le = preprocessing.LabelEncoder().fit(train.species)
    labels = le.transform(train.species)        # encode species strings
    classes = list(le.classes_)                 # save column names for submission
    test_ids = test.id                          # save test ids for submission

    train = train.drop(['species', 'id'], axis=1)
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes


if __name__ == "__main__":
    df_train = pd.read_csv('data/leaf-classification/train.csv')
    df_test = pd.read_csv('data/leaf-classification/test.csv')

    train, labels, test, test_ids, classes = encode(df_train, df_test)

    sss = model_selection.StratifiedShuffleSplit(10, test_size=0.2, random_state=23)
    for train_index, test_index in sss.split(df_train, labels):
        x_train, x_test = train.values[train_index], train.values[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.025, probability=True),
        NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        # GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    # Logging for Visual Comparison
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(x_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('****Results****')
        train_predictions = clf.predict(x_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        train_predictions = clf.predict_proba(x_test)
        ll = log_loss(y_test, train_predictions)
        print("Log Loss: {}".format(ll))

        log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
        log = log.append(log_entry)

    print("=" * 30)

    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()

    sns.set_color_codes("muted")
    sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

    plt.xlabel('Log Loss')
    plt.title('Classifier Log Loss')
    plt.show()

    # Predict Test Set
    favorite_clf = LinearDiscriminantAnalysis()
    favorite_clf.fit(x_train, y_train)
    test_predictions = favorite_clf.predict_proba(test)

    # Format DataFrame
    submission = pd.DataFrame(test_predictions, columns=classes)
    submission.insert(0, 'id', test_ids)
    submission.reset_index()

    # Export Submission
    submission.to_csv('data/leaf-classification/submission.csv', index=False)
    print(submission.tail())

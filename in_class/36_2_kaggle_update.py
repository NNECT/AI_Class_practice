import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing, model_selection
import seaborn as sns


def load_train():
    df = pd.read_csv('data/leaf-classification/train.csv', index_col=0)

    x = df.drop(['species'], axis=1)

    le = preprocessing.LabelEncoder().fit(df.species)
    y = le.transform(df.species)

    return x.values, y, le.classes_


def load_test():
    df = pd.read_csv('data/leaf-classification/test.csv', index_col=0)

    return df.values, df.index.values


if __name__ == "__main__":
    x_base, y_base, classes = load_train()
    x_test, y_test = load_test()

    data = model_selection.train_test_split(x_base, y_base, train_size=0.8)
    x_train, x_valid, y_train, y_valid = data

    # Classifiers
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, NuSVC
    from sklearn.tree import DecisionTreeClassifier
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

    log_cols = ["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)

    print('=' * 30)
    for clf in classifiers:
        clf.fit(x_train, y_train)
        clf.predict(x_valid)

        name = clf.__class__.__name__
        score = clf.score(x_valid, y_valid)
        print(name, score, sep='\n', end='\n' + '=' * 30 + '\n')

        log_entry = pd.DataFrame([[name, score]], columns=log_cols)
        log = log.append(log_entry)

    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()

    print(log['Accuracy'].argmax())

    # best_clf = classifiers[-2]
    # p = best_clf.predict_proba(x_test)
    #
    # with open('data/leaf-classification/submission.csv', 'w', encoding='utf-8') as f:
    #     f.write('id,' + ','.join(classes))
    #     for t_id, item in zip(y_test, p):
    #         f.write(str(t_id) + ','.join([str(v) for v in item]) + '\n')

    # submission = pd.DataFrame(p, columns=classes)
    # submission.insert(0, 'id', y_test)
    # submission.reset_index()

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE

def label_encode(df, columns):
    for col in columns:
        le = LabelEncoder()
        col_values_unique = list(df[col].unique())
        le_fitted = le.fit(col_values_unique)
        col_values = list(df[col].values)
        le.classes_
        col_values_transformed = le.transform(col_values)
        df[col] = col_values_transformed

def get_train_test(df, y_col, ratio):
    mask = np.random.rand(len(df)) &lt; ratio
    df_train = df[mask]
    df_test = df[~mask]
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    del df_train[y_col]
    del df_test[y_col]
    X_train = df_train.values
    X_test = df_test.values
    return X_train, Y_train, X_test, Y_test

def get_train_test(df, y_col, ratio):
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    del df_train[y_col]
    del df_test[y_col]
    X_train = df_train.values
    X_test = df_test.values
    return X_train, Y_train, X_test, Y_test

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators = 18),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB()
}

no_classifiers = len(dict_classifiers.keys())

def batch_classify(X_train, Y_train, X_test, Y_test, verbose = True):
    df_results = pandas.DataFrame(data=np.zeros(shape=(no_classifiers,4)), columns = ['classifier', 'train_score', 'test_score', 'training_time'])
    count = 0
    for key, classifier in dict_classifiers.items():
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        df_results.loc[count,'classifier'] = key
        df_results.loc[count,'train_score'] = train_score
        df_results.loc[count,'test_score'] = test_score
        df_results.loc[count,'training_time'] = t_diff
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=key, f=t_diff))
        count+=1
    return df_results

def get_unique_attribute(df):
    for col in df.columns.values:
        print(col, df[col].unique())

def main():
    # load dataset
    url_dataset = "script/mushrooms.csv"
    dataset = pandas.read_csv(url_dataset)
    # check the attibutes
    get_unique_attribute(dataset)
    #remove veil-type attribute
    del dataset['veil-type']
    #print(dataset.columns.values)
    # encode the attributes with LabelEncoder
    to_be_encoded_cols = dataset.columns.values
    label_encode(dataset, to_be_encoded_cols)
    # check the attibutes after encoded
    #get_unique_attribute(dataset)
    # split-out validation dataset
    array = dataset.values
    X = array[:,0:22]
    Y = array[:,0]
    validation_size = 0.20
    seed = 7
    scoring = "accuracy"
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('TREE', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    # feature extraction
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X, Y)
    # summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)
    # feature extraction
    model = LogisticRegression()
    rfe = RFE(model, 3)
    fit = rfe.fit(X, Y)
    print("Num Features: {}".format(fit.n_features_))
    print("Selected Features: {}".format(fit.support_))
    print("Feature Ranking: {}".format(fit.ranking_))
    # feature extraction
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)
    print("The Conclusion")
    print("Best Machine Learning Type for Mushroom Dataset is Binary Classification Model")
    print("because basicaly, the model is only divided to be two classes that are edible(e) and poisonous(p)")
    print("The Best Algorithm for Mushroom Dataset is DecisionTree, because DecisionTree is the most stable than others")
    print("If we change the seed,numbers of dataset itself whether increase or reduce,we can see that the accuracy of decision tree is highest and stable")
    print("As We can see at the result above(there are three feature selections/extractions), and we can pull the decision that Odor is the indicatived feature")
    print("The Most Importance or Indicative Feature/Attribute is Odor and it has big influence to predict whether the mushroom is edible/poisonous")
    print("Odor with value except Almond/None/Anise tend to be most indicatived attribute of poisonous mushroom")

if __name__ == '__main__':
    main()

from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas


def perform(classifiers, vectorizers, train_data, test_data):
    for classifier in classifiers:
      for vectorizer in vectorizers:
        string = ''
        string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

        # train
        vectorize_text = vectorizer.fit_transform(train_data.v2)
        classifier.fit(vectorize_text, train_data.v1)

        # score
        vectorize_text = vectorizer.transform(test_data.v2)
        score = classifier.score(vectorize_text, test_data.v1)
        string += '. Has score: ' + str(score)
        print(string)


data = pandas.read_csv('spam.csv', encoding='latin-1')
learn = data[:4400] # 4400 items
test = data[4400:] # 1172 items

perform(
    [
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        CalibratedClassifierCV(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        KNeighborsClassifier()
    ],
    [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer()
    ],
    learn,
    test
)


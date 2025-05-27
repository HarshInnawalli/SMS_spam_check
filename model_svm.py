from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from data_loader import load_data
from vectorizer import get_vectorizer

def run_svm():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    X_train, X_test, y_train, y_test = load_data(url)

    vectorizer = get_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LinearSVC(dual='auto')
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print("SVM Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_svm()

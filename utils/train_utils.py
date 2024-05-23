from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(X, y, X_test, seed=2024, return_model=False):
    model = LogisticRegression(random_state=seed, multi_class="ovr", n_jobs=1, max_iter=1000).fit(X, y)
    if return_model:
        return model, model.predict_proba(X_test).T if X_test is not None else []
    else:
        return model.predict_proba(X_test).T if X_test is not None else []
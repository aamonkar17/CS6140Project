"""
STACKING ENSEMBLE MODULE

Combines multiple base models using a meta-learner trained on out-of-fold
predictions. This allows the system to learn when each model performs best,
rather than relying on a single model globally.

In financial prediction tasks, different models capture different structures:
- Linear models capture stable relationships
- Tree models capture nonlinear interactions
Stacking combines both strengths.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

def stacking_ensemble(models, X_train, y_train, X_test):
    tscv = TimeSeriesSplit(n_splits=5)

    n_models = len(models)
    oof_preds = np.zeros((len(X_train), n_models))
    test_preds = np.zeros((len(X_test), n_models))

    for i, (name, model) in enumerate(models.items()):
        print(f"  Training base model: {name}")

        for tr_idx, val_idx in tscv.split(X_train):
            model.fit(X_train[tr_idx], y_train[tr_idx])
            oof_preds[val_idx, i] = model.predict(X_train[val_idx])

        # Fit on full data for test prediction
        model.fit(X_train, y_train)
        test_preds[:, i] = model.predict(X_test)

    # Meta-learner learns optimal combination
    meta = RidgeCV(alphas=np.logspace(-3, 3, 50))
    meta.fit(oof_preds, y_train)

    print("  Meta-learner trained (RidgeCV)")

    return meta.predict(test_preds)
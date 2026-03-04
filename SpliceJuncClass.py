import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier



DATA_PATH = r"C:\Users\hersc\Downloads\molecular+biology+splice+junction+gene+sequences\splice.data"
OUTDIR = "outputs_splice"
os.makedirs(OUTDIR, exist_ok=True)

RANDOM_STATE = 42



def save_confusion_matrix(y_true, y_pred, labels, title, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap=None, colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def summarize_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
    }


def decode_onehot_feature_name(feat_name: str):
    """
    Encoder feature names look like:
      x0_A, x1_G, x12_T, etc (depends on sklearn version).
    We convert to: (position, base)
    """
    m = re.match(r"^x?(\d+)_(.+)$", feat_name)
    if not m:
        return None, feat_name
    pos = int(m.group(1))
    base = m.group(2)
    return pos, base


def plot_permutation_importance(result, feature_names, top_n, title, outpath):
    importances = result.importances_mean
    idx = np.argsort(importances)[::-1][:top_n]

    top_feats = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    plt.figure(figsize=(9, 6))
    plt.barh(range(len(top_feats))[::-1], top_vals[::-1])
    plt.yticks(range(len(top_feats))[::-1], top_feats[::-1])
    plt.xlabel("Permutation importance (mean decrease in score)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


print("[INFO] Loading splice.data...")
df = pd.read_csv(DATA_PATH, header=None)
df.columns = ["class", "instance", "sequence"]

df["sequence"] = df["sequence"].astype(str).str.replace(" ", "")

print(df.head())

sequence_df = df["sequence"].apply(list)
sequence_df = pd.DataFrame(sequence_df.tolist())

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

X = encoder.fit_transform(sequence_df)
X = X.astype(np.float64)

if not np.isfinite(X).all():
    raise ValueError("X contains NaN or inf values.")

y = df["class"].astype(str)


try:
    feature_names = encoder.get_feature_names_out(sequence_df.columns)
except Exception:
    # fallback
    feature_names = np.array([f"f{i}" for i in range(X.shape[1])])

print("[INFO] Feature shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

labels = sorted(y.unique())


baseline_models = {
    "Naive Bayes (Gaussian)": GaussianNB(),
}


tuned_models = {
    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [1, 3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski"],  
        }
    ),
    "SVM": (
        SVC(),
        {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10],
            "gamma": ["scale"],  
        }
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        }
    ),
    "Neural Network (MLP)": (
        MLPClassifier(
            max_iter=1000,
            random_state=RANDOM_STATE,
            early_stopping=False
        ),
        {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.001],   
            "solver": ["adam"],              
        }
    ),
}

SCORING = "f1_macro"  



all_results = []
best_estimators = {}

print("\n[INFO] Training baseline models...")
for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = summarize_metrics(y_test, preds)
    metrics["model"] = name
    metrics["best_params"] = "(baseline)"
    all_results.append(metrics)

    save_confusion_matrix(
        y_test, preds, labels,
        title=f"{name} Confusion Matrix",
        outpath=os.path.join(OUTDIR, f"cm_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    )

    print(f"[RESULT] {name} accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

print("\n[INFO] Tuning models with GridSearchCV...")
for name, (estimator, param_grid) in tuned_models.items():
    print(f"\n[TUNING] {name} ...")
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=SCORING,
        cv=5,
        n_jobs=-1,
        verbose=0,
        error_score=np.nan
    )
    grid.fit(X_train, y_train)
    
    if np.isnan(grid.best_score_):
        print(f"[WARN] {name} tuning failed for all params; skipping.")
        continue

    best = grid.best_estimator_
    best_estimators[name] = best

    preds = best.predict(X_test)
    metrics = summarize_metrics(y_test, preds)
    metrics["model"] = name
    metrics["best_params"] = str(grid.best_params_)
    all_results.append(metrics)

   
    save_confusion_matrix(
        y_test, preds, labels,
        title=f"{name} Confusion Matrix",
        outpath=os.path.join(OUTDIR, f"cm_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    )

   
    report_txt = classification_report(y_test, preds, digits=4, zero_division=0)
    with open(os.path.join(OUTDIR, f"report_{name.replace(' ', '_').replace('(', '').replace(')', '')}.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"[BEST] {name}: {grid.best_params_}")
    print(f"[RESULT] {name} accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

    
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df.sort_values("mean_test_score", ascending=False).head(25).to_csv(
        os.path.join(OUTDIR, f"cv_top25_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"),
        index=False
    )


results_df = pd.DataFrame(all_results).sort_values("f1_macro", ascending=False)
results_df.to_csv(os.path.join(OUTDIR, "results_summary.csv"), index=False)

print("\n[FINAL RESULTS - sorted by f1_macro]")
print(results_df[["model", "accuracy", "f1_macro", "precision_macro", "recall_macro", "best_params"]])


plt.figure(figsize=(9, 5))
plt.bar(results_df["model"], results_df["accuracy"])
plt.xticks(rotation=25, ha="right")
plt.ylabel("Accuracy")
plt.title("Classifier Accuracy (test set)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "accuracy_comparison.png"), dpi=200)
plt.close()


plt.figure(figsize=(9, 5))
plt.bar(results_df["model"], results_df["f1_macro"])
plt.xticks(rotation=25, ha="right")
plt.ylabel("F1 (macro)")
plt.title("Classifier F1 Macro (test set)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "f1_macro_comparison.png"), dpi=200)
plt.close()

print(f"\n[INFO] Saved summary tables + plots to: {OUTDIR}")



best_model_name = results_df.iloc[0]["model"]
print(f"\n[INFO] Best model by f1_macro: {best_model_name}")

models_for_importance = []


if best_model_name in best_estimators:
    models_for_importance.append((best_model_name, best_estimators[best_model_name]))


if "Decision Tree" in best_estimators:
    models_for_importance.append(("Decision Tree", best_estimators["Decision Tree"]))

for model_name, model in models_for_importance:
    perm = permutation_importance(
        model, X_test, y_test,
        scoring="f1_macro",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    
    readable_names = []
    for fn in feature_names:
        pos, base = decode_onehot_feature_name(str(fn))
        if pos is None:
            readable_names.append(str(fn))
        else:
            readable_names.append(f"pos{pos}={base}")

  
    outpath = os.path.join(OUTDIR, f"perm_importance_top30_{model_name.replace(' ', '_')}.png")
    plot_permutation_importance(
        perm, readable_names, top_n=30,
        title=f"Permutation Importance (Top 30) - {model_name}",
        outpath=outpath
    )

    
    imp_df = pd.DataFrame({
        "feature": readable_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)

    imp_df.head(100).to_csv(
        os.path.join(OUTDIR, f"perm_importance_top100_{model_name.replace(' ', '_')}.csv"),
        index=False
    )

print("\n[DONE] Confusion matrices, metrics, tuning, and permutation importance complete.")
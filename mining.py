"""
mining.py - data preparation and data mining functions for the
ADSSU Classroom Facility Issue Dashboard.

Techniques:
  1. Text classification  (TF-IDF + Logistic Regression / Naive Bayes / Linear SVM)
  2. Association rule mining (Apriori, implemented with pandas + itertools)
  3. Rule-based maintenance prioritization (weighted priority score)
"""
import re
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, classification_report)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

REQUIRED_COLUMNS = [
    "report_id", "timestamp", "campus_location", "building_name", "room_number",
    "reporter_role", "issue_description", "category", "urgency_level",
    "resolution_status", "days_to_resolve",
]
URGENCY_ORDER = ["Low", "Medium", "High", "Critical"]
URGENCY_WEIGHT = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# keyword -> facility item (used to turn free text into "items" for association rules)
ITEM_PATTERNS = {
    "Light": r"\b(lights?|bulbs?|fluorescent)\b",
    "Ceiling Fan": r"\b(fans?|fan regulator)\b",
    "Outlet": r"\b(outlets?|sockets?|extension cord)\b",
    "Switch": r"\bswitch",
    "Wiring": r"\b(wiring|wires?|breaker)\b",
    "Aircon": r"(aircon|air[- ]?con|air conditioner)",
    "Ventilation": r"(ventilation|air vent|louver)",
    "Chair": r"\b(chairs?|armchairs?)\b",
    "Table": r"\b(tables?|desks?)\b",
    "Whiteboard": r"(whiteboard|eraser tray)",
    "Chalkboard": r"(chalkboard|blackboard)",
    "Podium": r"podium",
    "Projector": r"projector",
    "Smart TV": r"(smart tv|\btv\b)",
    "Wi-Fi": r"(wi-?fi|access point)",
    "Speaker": r"(speaker|microphone)",
    "Computer": r"(computer|monitor)",
    "Faucet": r"faucet",
    "Sink": r"\bsink",
    "Drain": r"drain",
    "Water Pipe": r"\bpipe",
    "Door": r"(door|doorknob)",
    "Window": r"(window|jalousie)",
    "Ceiling": r"ceiling(?! fan)",
    "Roof": r"\broof",
    "Floor": r"floor(?! drain)",
}
_COMPILED = {k: re.compile(v, re.I) for k, v in ITEM_PATTERNS.items()}

CATEGORY_ACTIONS = {
    "Electrical": "Have the campus electrician inspect wiring, outlets and lighting; replace faulty parts and check for fire risk.",
    "HVAC/Ventilation": "Schedule preventive aircon servicing (cleaning, drain check, gas level) and fix ventilation blockages.",
    "Furniture & Fixtures": "Repair or replace damaged furniture and boards; run a furniture audit of the affected rooms.",
    "IT & Audio-Visual": "Coordinate with the IT office to check projectors, displays, speakers and Wi-Fi access points.",
    "Plumbing": "Request plumbing inspection for leaks and clogged drains; prioritize leaks that damage floors or electrical parts.",
    "Structural/Civil": "Request a civil/structural inspection; fix roof, ceiling and window leaks before the rainy season.",
}


# ---------------------------------------------------------------- data prep
def validate_columns(df):
    """Return the list of required columns that are missing."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def prepare_data(df):
    """Clean types and add helper columns."""
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["days_to_resolve"] = pd.to_numeric(df["days_to_resolve"], errors="coerce")
    df["urgency_level"] = pd.Categorical(df["urgency_level"], URGENCY_ORDER, ordered=True)
    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["week"] = df["timestamp"].dt.to_period("W").dt.start_time
    df["is_open"] = (df["resolution_status"] != "Resolved").astype(int)
    df["items"] = df["issue_description"].apply(extract_items)
    return df


def extract_items(text):
    """Find which facility items a report mentions (keyword matching)."""
    found = [name for name, rx in _COMPILED.items() if rx.search(str(text))]
    return found


# ------------------------------------------------------ text classification
def _make_model(name):
    if name == "Naive Bayes":
        clf = MultinomialNB()
    elif name == "Linear SVM":
        clf = LinearSVC()
    else:
        clf = LogisticRegression(max_iter=2000)
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True), clf)


def train_classifier(df, model_name="Logistic Regression", test_size=0.2, seed=42):
    """Train a text classifier that predicts the issue category from the description."""
    data = df[["issue_description", "category"]].dropna()
    X, y = data["issue_description"], data["category"]
    counts = y.value_counts()
    strat = y if counts.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)

    model = _make_model(model_name).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    labels = sorted(y.unique())

    prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    report = pd.DataFrame(classification_report(y_te, pred, output_dict=True, zero_division=0)).T.round(3)
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels)

    # k-fold cross-validation (more reliable than a single split)
    k = int(min(5, counts.min())) if counts.min() >= 2 else 0
    cv = None
    if k >= 2:
        res = cross_validate(_make_model(model_name), X, y,
                             cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                             scoring=["accuracy", "f1_macro"])
        cv = {"folds": k,
              "accuracy_mean": float(res["test_accuracy"].mean()), "accuracy_std": float(res["test_accuracy"].std()),
              "f1_mean": float(res["test_f1_macro"].mean()), "f1_std": float(res["test_f1_macro"].std())}
    return {
        "model": model, "model_name": model_name, "labels": labels,
        "n_train": len(X_tr), "n_test": len(X_te),
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "report": report, "confusion": cm, "cv": cv,
        "top_terms": top_terms(model, labels),
    }


def top_terms(model, labels, n=6):
    """Most influential words per category (for interpretation)."""
    vec, clf = model.steps[0][1], model.steps[1][1]
    terms = np.array(vec.get_feature_names_out())
    if hasattr(clf, "coef_"):
        w = clf.coef_
        if w.shape[0] == 1 and len(clf.classes_) == 2:   # binary case
            w = np.vstack([-w[0], w[0]])
    elif hasattr(clf, "feature_log_prob_"):
        w = clf.feature_log_prob_
    else:
        return {}
    return {c: list(terms[np.argsort(w[i])[::-1][:n]]) for i, c in enumerate(clf.classes_)}


# --------------------------------------------------- association rule mining
def build_baskets(df, basket_by="Room + Month"):
    """One basket = the set of items reported together (same room, optionally same month)."""
    keys = {"Room": ["room_number"], "Room + Month": ["room_number", "month"], "Report": ["report_id"]}[basket_by]
    ex = df[keys + ["items"]].explode("items").dropna(subset=["items"]).reset_index(drop=True)
    return pd.crosstab([ex[k] for k in keys], ex["items"]).clip(upper=1).astype(bool)


def apriori_rules(baskets, min_support=0.1, min_confidence=0.5, min_lift=1.0, max_len=3):
    """Simple Apriori: frequent itemsets (size<=max_len) -> rules with one-item consequent."""
    n = len(baskets)
    if n == 0:
        return pd.DataFrame()
    cols = list(baskets.columns)
    arr = baskets.values
    idx = {c: i for i, c in enumerate(cols)}

    def support(items):
        return arr[:, [idx[i] for i in items]].all(axis=1).sum() / n

    freq = {}
    level = [(c,) for c in cols if support((c,)) >= min_support]
    for t in level:
        freq[t] = support(t)
    size = 1
    while level and size < max_len:
        size += 1
        items = sorted({i for t in level for i in t})
        cands = [c for c in combinations(items, size)
                 if all(sub in freq for sub in combinations(c, size - 1))]
        level = []
        for c in cands:
            s = support(c)
            if s >= min_support:
                freq[c] = s
                level.append(c)
    rows = []
    for itemset, s in freq.items():
        if len(itemset) < 2:
            continue
        for cons in itemset:
            ante = tuple(i for i in itemset if i != cons)
            conf = s / freq[ante]
            lift = conf / freq[(cons,)]
            if conf >= min_confidence and lift >= min_lift:
                rows.append({"if_reported": ", ".join(ante), "then_also_reported": cons,
                             "support": round(s, 3), "confidence": round(conf, 3), "lift": round(lift, 2),
                             "baskets": int(round(s * n))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)


def explain_rule(row):
    return (f"When **{row['if_reported']}** is reported, **{row['then_also_reported']}** is also reported "
            f"in {row['confidence']:.0%} of cases ({row['lift']:.1f}x more often than by chance).")


# ------------------------------------------------ maintenance prioritization
def _norm(s):
    s = s.astype(float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0


def priority_table(df, by=("building_name", "room_number"), weights=(0.35, 0.30, 0.20, 0.15)):
    """Weighted priority score per room (0-100).
    weights = (urgency points, open reports, report frequency, average days to resolve)"""
    by = list(by)
    d = df.copy()
    d["w"] = d["urgency_level"].astype(str).map(URGENCY_WEIGHT).fillna(1)
    g = d.groupby(by).agg(reports=("report_id", "count"), urgency_points=("w", "sum"),
                          open_reports=("is_open", "sum"), avg_days=("days_to_resolve", "mean"),
                          critical=("urgency_level", lambda s: int((s.astype(str) == "Critical").sum())),
                          main_category=("category", lambda s: s.value_counts().idxmax())).reset_index()
    g["avg_days"] = g["avg_days"].fillna(0).round(1)
    wu, wo, wf, wd = weights
    g["priority_score"] = (100 * (wu * _norm(g["urgency_points"]) + wo * _norm(g["open_reports"])
                                  + wf * _norm(g["reports"]) + wd * _norm(g["avg_days"]))).round(1)
    g["priority_level"] = pd.cut(g["priority_score"], [-0.1, 40, 70, 100.1],
                                 labels=["Routine", "Schedule soon", "Immediate"])
    g["recommended_action"] = g["main_category"].map(CATEGORY_ACTIONS)
    return g.sort_values("priority_score", ascending=False).reset_index(drop=True)


def build_report_text(df, prio, clf=None, rules=None, title="Classroom Facility Issue Summary"):
    """Plain-text summary for decision makers (download)."""
    lines = [title, "=" * len(title), ""]
    lines.append(f"Reports analysed: {len(df)}  |  Period: {df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}")
    lines.append(f"Open (not yet resolved): {int(df['is_open'].sum())}  |  Critical reports: {int((df['urgency_level'].astype(str) == 'Critical').sum())}")
    avg = df["days_to_resolve"].mean()
    if pd.notna(avg):
        lines.append(f"Average days to resolve: {avg:.1f}")
    lines += ["", "Reports by category:"]
    for k, v in df["category"].value_counts().items():
        lines.append(f"  - {k}: {v}")
    lines += ["", "Top 5 priority rooms:"]
    for _, r in prio.head(5).iterrows():
        lines.append(f"  - {r['room_number']} ({r['building_name']}): score {r['priority_score']}, "
                     f"{r['reports']} reports, {r['open_reports']} open, main issue: {r['main_category']}")
        lines.append(f"      Action: {r['recommended_action']}")
    if clf is not None:
        lines += ["", f"Text classification ({clf['model_name']}): accuracy {clf['accuracy']:.1%}, macro F1 {clf['f1']:.3f}"]
    if rules is not None and len(rules):
        lines += ["", "Top association rules:"]
        for _, r in rules.head(5).iterrows():
            lines.append(f"  - {r['if_reported']} -> {r['then_also_reported']} (conf {r['confidence']:.0%}, lift {r['lift']})")
    lines += ["", "Note: results are decision-support evidence, not a replacement for ocular inspection."]
    return "\n".join(lines)

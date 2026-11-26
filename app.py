import pandas as pd
import plotly.express as px
import streamlit as st

import mining as m

st.set_page_config(page_title="ADSSU Classroom Facility Issue Dashboard", page_icon="🏫", layout="wide")

DATA_PATH = "data/classroom_issues.csv"


# ------------------------------------------------------------------ data
@st.cache_data
def load_default():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def prepare(raw):
    return m.prepare_data(raw)


@st.cache_data
def run_classifier(df, model_name):
    return m.train_classifier(df, model_name)


@st.cache_data
def run_rules(df, basket_by, min_support, min_conf, min_lift):
    baskets = m.build_baskets(df, basket_by)
    return len(baskets), m.apriori_rules(baskets, min_support, min_conf, min_lift)


st.title("🏫 Classroom Facility Issue Dashboard")
st.caption("Classroom Facility Issue Pattern Mining and Maintenance Prioritization Using Text Classification "
           "and Association Rule Mining — Agusan del Sur State University")

# sidebar: data source
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source = "Uploaded file"
else:
    raw = load_default()
    source = "Sample dataset for prototype purposes"

missing = m.validate_columns(raw)
if missing:
    st.error(f"The CSV is missing required columns: {', '.join(missing)}")
    st.stop()

full = prepare(raw)
st.sidebar.caption(f"Source: {source}")
if source.startswith("Sample"):
    st.info("This dashboard is using a **sample dataset for prototype purposes**. "
            "Replace `data/classroom_issues.csv` with real ADSSU records before drawing conclusions.")

# sidebar: filters
st.sidebar.header("Filters")
colleges = sorted(full["building_name"].unique())
sel_col = st.sidebar.multiselect("College / Building", colleges, colleges)
cats = sorted(full["category"].unique())
sel_cat = st.sidebar.multiselect("Category", cats, cats)
urg = [u for u in m.URGENCY_ORDER if u in set(full["urgency_level"].astype(str))]
sel_urg = st.sidebar.multiselect("Urgency level", urg, urg)
status_opts = sorted(full["resolution_status"].unique())
sel_status = st.sidebar.multiselect("Status", status_opts, status_opts)
dmin, dmax = full["timestamp"].min().date(), full["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)

df = full[full["building_name"].isin(sel_col) & full["category"].isin(sel_cat)
          & full["urgency_level"].astype(str).isin(sel_urg) & full["resolution_status"].isin(sel_status)]
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    df = df[(df["timestamp"].dt.date >= date_range[0]) & (df["timestamp"].dt.date <= date_range[1])]

if df.empty:
    st.warning("No records match the current filters.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1 Data Exploration", "2 Text Classification", "3 Association Rules", "4 Decision Support", "5 Report"])

# ----------------------------------------------------------- 1. exploration
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reports", len(df))
    c2.metric("Open reports", int(df["is_open"].sum()))
    c3.metric("Critical reports", int((df["urgency_level"].astype(str) == "Critical").sum()))
    c4.metric("Avg days to resolve", f"{df['days_to_resolve'].mean():.1f}")

    a, b = st.columns(2)
    with a:
        cc = df["category"].value_counts().reset_index()
        cc.columns = ["category", "reports"]
        st.plotly_chart(px.bar(cc, x="reports", y="category", orientation="h", title="Reports by category"))
    with b:
        uu = df.groupby(["building_name", "urgency_level"], observed=True).size().reset_index(name="reports")
        st.plotly_chart(px.bar(uu, x="building_name", y="reports", color="urgency_level",
                               category_orders={"urgency_level": m.URGENCY_ORDER},
                               title="Reports by college and urgency"))
    a, b = st.columns(2)
    with a:
        wk = df.groupby("week").size().reset_index(name="reports")
        st.plotly_chart(px.line(wk, x="week", y="reports", markers=True, title="Weekly number of reports"))
    with b:
        ad = df.groupby("category")["days_to_resolve"].mean().round(1).reset_index()
        st.plotly_chart(px.bar(ad, x="category", y="days_to_resolve", title="Average days to resolve by category"))

    heat = pd.crosstab(df["building_name"], df["category"])
    st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto", title="Heatmap: college vs. category"))

    with st.expander("Dataset overview, summary statistics and missing values"):
        st.write(f"**{len(df)}** records (filtered) from **{len(full)}** total, "
                 f"covering {df['timestamp'].min():%b %d, %Y} to {df['timestamp'].max():%b %d, %Y}.")
        st.dataframe(df[["days_to_resolve"]].describe().T)
        miss = raw.isna().sum().reset_index()
        miss.columns = ["column", "missing_values"]
        st.dataframe(miss)
        st.caption("Blank `days_to_resolve` is expected for reports that are still Pending or In Progress.")

    st.subheader("Drill-down")
    col_pick = st.selectbox("Choose a college", sorted(df["building_name"].unique()))
    sub = df[df["building_name"] == col_pick]
    room_pick = st.selectbox("Choose a room", ["All rooms"] + sorted(sub["room_number"].unique()))
    if room_pick != "All rooms":
        sub = sub[sub["room_number"] == room_pick]
    st.dataframe(sub[["report_id", "timestamp", "room_number", "reporter_role", "issue_description",
                      "category", "urgency_level", "resolution_status", "days_to_resolve"]]
                 .sort_values("timestamp", ascending=False))

# ------------------------------------------------------ 2. classification
with tab2:
    st.markdown("**Goal:** automatically assign a category to a new issue report from its text, "
                "so reports reach the right maintenance unit faster.")
    model_name = st.selectbox("Algorithm", ["Logistic Regression", "Linear SVM", "Naive Bayes"])
    res = run_classifier(full, model_name)   # trained on the full dataset (not the filtered view)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy (test set)", f"{res['accuracy']:.1%}")
    c2.metric("Macro precision", f"{res['precision']:.3f}")
    c3.metric("Macro recall", f"{res['recall']:.3f}")
    c4.metric("Macro F1", f"{res['f1']:.3f}")
    st.caption(f"Train/test split: {res['n_train']} / {res['n_test']} reports (80/20, stratified).")
    if res["cv"]:
        cv = res["cv"]
        st.write(f"**{cv['folds']}-fold cross-validation:** accuracy {cv['accuracy_mean']:.1%} "
                 f"(± {cv['accuracy_std']:.1%}), macro F1 {cv['f1_mean']:.3f} (± {cv['f1_std']:.3f})")

    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.imshow(res["confusion"], text_auto=True, aspect="auto",
                                  labels=dict(x="Predicted", y="Actual"), title="Confusion matrix (test set)"))
    with b:
        st.write("**Per-category results**")
        st.dataframe(res["report"])

    st.subheader("Interpretation: words that signal each category")
    st.dataframe(pd.DataFrame({k: ", ".join(v) for k, v in res["top_terms"].items()}.items(),
                              columns=["Category", "Most influential words"]))

    st.subheader("Try it: classify a new report")
    txt = st.text_area("Issue description", "Projector shows no signal and the Wi-Fi keeps disconnecting")
    if txt.strip():
        model = res["model"]
        pred = model.predict([txt])[0]
        st.success(f"Predicted category: **{pred}**")
        st.write("Suggested action:", m.CATEGORY_ACTIONS.get(pred, "-"))
    st.warning("If the sample dataset is used, accuracy will look very high because the descriptions are "
               "template-based. Expect lower, more realistic scores with real reports.")

# ------------------------------------------------------ 3. association rules
with tab3:
    st.markdown("**Goal:** discover which facility problems tend to appear together in the same room, "
                "so they can be inspected and repaired in one visit.")
    c1, c2, c3, c4 = st.columns(4)
    basket_by = c1.selectbox("A basket is…", ["Room + Month", "Room", "Report"],
                             help="Items reported together in the same room and month (or room overall / single report).")
    min_sup = c2.slider("Min support", 0.01, 0.5, 0.10, 0.01)
    min_conf = c3.slider("Min confidence", 0.1, 1.0, 0.5, 0.05)
    min_lift = c4.slider("Min lift", 1.0, 5.0, 1.2, 0.1)
    n_baskets, rules = run_rules(df, basket_by, min_sup, min_conf, min_lift)
    st.caption(f"{n_baskets} baskets analysed. Facility items are extracted from the issue descriptions by keyword matching.")
    if rules.empty:
        st.info("No rules found. Lower the minimum support, confidence or lift.")
    else:
        top = rules.head(25)
        st.dataframe(top)
        st.plotly_chart(px.scatter(top, x="support", y="confidence", size="lift", color="lift",
                                   hover_data=["if_reported", "then_also_reported"],
                                   title="Rules: support vs. confidence (bubble size = lift)"))
        st.subheader("Interpretation of the strongest rules")
        for _, r in top.head(5).iterrows():
            st.markdown("- " + m.explain_rule(r))
        st.caption("Support = share of baskets containing the items; confidence = how often the second item "
                   "appears when the first does; lift > 1 means the items appear together more than by chance.")

# ----------------------------------------------------- 4. decision support
with tab4:
    st.markdown("**Goal:** rank rooms by maintenance priority and recommend actions.")
    st.write("Priority score (0–100) = weighted combination of urgency points, open reports, number of reports, "
             "and average days to resolve. Adjust the weights below.")
    w1, w2, w3, w4 = st.columns(4)
    wu = w1.slider("Urgency", 0.0, 1.0, 0.35, 0.05)
    wo = w2.slider("Open reports", 0.0, 1.0, 0.30, 0.05)
    wf = w3.slider("Report frequency", 0.0, 1.0, 0.20, 0.05)
    wd = w4.slider("Avg days to resolve", 0.0, 1.0, 0.15, 0.05)
    tot = wu + wo + wf + wd
    weights = (wu / tot, wo / tot, wf / tot, wd / tot) if tot > 0 else (0.35, 0.30, 0.20, 0.15)
    prio = m.priority_table(df, weights=weights)

    levels = prio["priority_level"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Immediate", int(levels.get("Immediate", 0)))
    c2.metric("Schedule soon", int(levels.get("Schedule soon", 0)))
    c3.metric("Routine", int(levels.get("Routine", 0)))

    if len(prio) > 5:
        top_n = st.slider("Show top N rooms", 5, min(30, len(prio)), min(10, len(prio)))
    else:
        top_n = len(prio)
    st.plotly_chart(px.bar(prio.head(top_n), x="priority_score", y="room_number", color="priority_level",
                           orientation="h", hover_data=["building_name", "reports", "open_reports", "main_category"],
                           title="Top priority rooms").update_yaxes(autorange="reversed"))
    st.dataframe(prio.head(top_n)[["room_number", "building_name", "reports", "open_reports", "critical",
                                   "avg_days", "main_category", "priority_score", "priority_level",
                                   "recommended_action"]])

    st.subheader("Recommended management actions")
    for _, r in prio.head(5).iterrows():
        st.markdown(f"- **{r['room_number']}** ({r['building_name']}), score {r['priority_score']} — "
                    f"{r['reports']} reports, {r['open_reports']} still open. "
                    f"Main issue: *{r['main_category']}*. {r['recommended_action']}")

    cat_prio = (df.groupby("category").agg(reports=("report_id", "count"), avg_days=("days_to_resolve", "mean"),
                                           open_reports=("is_open", "sum")).round(1)
                .sort_values("reports", ascending=False).reset_index())
    st.subheader("Category-level planning")
    st.dataframe(cat_prio)
    st.caption("Use these results as evidence for budgeting, scheduling preventive maintenance, "
               "and deciding which rooms to inspect first.")

# ---------------------------------------------------------------- 5. report
with tab5:
    st.markdown("Download summaries for decision-makers (based on the current filters).")
    clf_res = run_classifier(full, "Logistic Regression")
    _, rep_rules = run_rules(df, "Room + Month", 0.10, 0.5, 1.2)
    prio_default = m.priority_table(df)
    text = m.build_report_text(df, prio_default, clf_res, rep_rules)
    st.text(text)
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ Summary report (.txt)", text, "classroom_issue_summary.txt")
    c2.download_button("⬇ Priority rooms (.csv)", prio_default.to_csv(index=False), "priority_rooms.csv")
    c3.download_button("⬇ Filtered data (.csv)",
                       df.drop(columns=["items", "date", "month", "week", "is_open"]).to_csv(index=False),
                       "filtered_reports.csv")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score
from plot_results import plot
import pandas as pd
from dirty_logs import *
from Declare4Py.Encodings.Aggregate import Aggregate
from Declare4Py.Encodings.IndexBased import IndexBased
from Declare4Py.Encodings.LastState import LastState
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

#CLASSIFICATION_ALGORITHMS = ["Gradient Boosting"]
#ENCODINGS = ["Aggregation"]

#fare ciclo for con tutte e 8 le funzioni di sporcaggio, provare con 6 algoritmi e provare anche con un dataset esterno

CLASSIFICATION_ALGORITHMS = ["Random Forest","Gradient Boosting"]
ENCODINGS = ["Aggregation","Last State","Index-based"]
DIRTY_FUNCTIONS = ["Same Label different Activities"]#["Missing Timestamp","Missing CaseId","Missing Activities","Missing Events","Wrong Timestamp","Wrong Activities","Same Timestamp different Events","Same Label different Activities","Irrelevant Events"]

def classification(X, y, classifier, seed):
    X = X.astype(int)
    y = y.astype(int)

    X = np.nan_to_num(X)
    clf = RandomForestClassifier()

    if classifier == "Random Forest":
        clf = RandomForestClassifier()
    elif classifier == "Gradient Boosting":
        clf = GradientBoostingClassifier()

    print("Training for "+classifier+"...")

    model_fit = clf.fit(X, y)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    #clf.fit(X_train, y_train)
    #y_pred = clf.predict(X_test)

    cv = ShuffleSplit(n_splits=8, test_size=0.3, random_state=seed)

    model_scores_1 = cross_val_score(model_fit, X, y, cv=cv, scoring="precision_weighted")
    model_scores_2 = cross_val_score(model_fit, X, y, cv=cv, scoring="recall_weighted")
    model_scores_3 = cross_val_score(model_fit, X, y, cv=cv, scoring="f1_weighted")

    precision = model_scores_1.mean()
    recall = model_scores_2.mean()
    f1 = model_scores_3.mean()

    #precision = precision_score(y_test, y_pred, average='weighted')
    #recall = recall_score(y_test, y_pred, average='weighted')
    #f1 = f1_score(y_test, y_pred, average='weighted')

    print("Precision: "+str(precision))
    print("Recall: "+str(recall))
    print("F1: "+str(f1))

    return {"precision": precision,
            "recall": recall,
            "f1": f1}


if __name__ == '__main__':

    for dirty_f in DIRTY_FUNCTIONS:
        results_for_each_algorithm = []
        for algorithm in CLASSIFICATION_ALGORITHMS:  # FIRST CICLE ON THE ALGORITHMS
            for e in ENCODINGS:

                results_single_algorithm = []

                # DATA COLLECTION
                df = pd.read_csv("logs.csv")

                # DATA POLLUTION
                if dirty_f == "Missing Timestamp":
                    data = inject_missing(df, "Timestamp", 1)
                elif dirty_f == "Missing CaseId":
                    data = inject_missing(df, "CaseId", 1)
                elif dirty_f == "Missing Activities":
                    data = inject_missing(df, "WorkflowModelElement", 1)
                elif dirty_f == "Missing Events":
                    data = remove_events(df, 1)
                elif dirty_f == "Wrong Timestamp":
                    data = wrong_timestamp(df, "Timestamp", 3, 1)
                elif dirty_f == "Wrong Activities":
                    data = wrong_event(df, "WorkflowModelElement", 1, 1)
                elif dirty_f == "Same Timestamp different Events":
                    data = same_timestamp_different_events(df, "Timestamp", 1)
                elif dirty_f == "Same Label different Activities":
                    data = same_label_different_activities(df, "WorkflowModelElement", "terminate_exchange", 5)
                else:
                    data = irrelevant_events(df, 1, 1)

                for d in data:  # SECOND CICLE ON THE NUMBER OF POLLUTED DATASET WITH DIFFERENT % OF POLLUTION

                    if dirty_f in ["Missing Timestamp","Wrong Timestamp","Same Timestamp different Events","Irrelevant Events"]:
                        d = d.sort_values(by=['CaseId', 'Timestamp'])

                    # ENCODING AND DATA ANALYSIS
                    if e == "Aggregation":
                        variants_discovery = Pipeline([('vect', Aggregate(case_id_col="CaseId", cat_cols=["WorkflowModelElement"],
                                                                  num_cols=[], boolean=True))])
                    elif e == "Last State":
                        variants_discovery = Pipeline([('vect', LastState(case_id_col="CaseId", cat_cols=["WorkflowModelElement"], num_cols=[]))])
                    else:
                        variants_discovery = Pipeline([('vect', IndexBased(case_id_col="CaseId", cat_cols=["WorkflowModelElement"], num_cols=[], create_dummies=True))])

                    df = variants_discovery.fit_transform(d)

                    target = "WorkflowModelElement_terminate_exchange"

                    if e == "Index-based":
                        dataset_target = pd.DataFrame([])
                        for c in df.columns:
                            if c.endswith("terminate_exchange"):
                                dataset_target[c] = df[c]
                                df = df.drop(columns=c)
                        dataset_target["Outcome"] = dataset_target.any(axis='columns')
                        df["WorkflowModelElement_terminate_exchange"] = dataset_target["Outcome"]

                    columns = df.columns.drop(target)
                    X = df[columns]
                    y = df[target]

                    results_1_analysis = classification(X, y, algorithm, 1)
                    results_single_algorithm.append(results_1_analysis)

                results_for_each_algorithm.append(results_single_algorithm)

        with open("results_project_BPM_"+dirty_f+".txt", "w") as file:
            file.write(str(results_for_each_algorithm))
            file.close()

        if dirty_f in ["Missing Timestamp","Missing CaseId","Missing Activities","Missing Events"]:
            xlabel = "Completeness"
        elif dirty_f in ["Wrong Timestamp", "Wrong Activities", "Same Timestamp different Events"]:
            xlabel = "Accuracy"
        elif dirty_f in ["Same Label different Activities"]:
            xlabel = "Consistency"
        else:
            xlabel = "Duplication"

        # RESULTS EVALUATION
        plot(x_axis_values=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1],
             x_label=xlabel, results=results_for_each_algorithm, title=dirty_f+" (f1)",
             algorithms=CLASSIFICATION_ALGORITHMS, encodings=ENCODINGS, plot_type="f1")
        plot(x_axis_values=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1],
             x_label=xlabel, results=results_for_each_algorithm, title=dirty_f+" (precision)",
             algorithms=CLASSIFICATION_ALGORITHMS, encodings=ENCODINGS, plot_type="precision")
        plot(x_axis_values=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1],
             x_label=xlabel, results=results_for_each_algorithm, title=dirty_f+" (recall)",
             algorithms=CLASSIFICATION_ALGORITHMS, encodings=ENCODINGS, plot_type="recall")

"""This file is part of ASKF-Bench.

ASKF-Bench is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

ASKF-Bench is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with ASKF-Bench.
If not, see <https://www.gnu.org/licenses/>.
"""

import datetime
import time
import sys
import json
import h5py
import scipy.io
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split


def load_kernels(fname, kernel_table, label_table):
    """Extract kernels and labels from mat5 or mat7 file.

    Parameters
    ----------
    fname, name of the matrix file
    kernel_table, table containing the list of kernel table names
    label_table, name of the target table

    Return
    ------
    kernel list
    target array
    """
    try:
        mat = scipy.io.loadmat(fname)
        klist = []
        for k in mat[kernel_table]:
            klist.append(np.array(k[0], dtype=np.double))
        labels = mat[label_table].astype(int)
        labels = labels - np.min(labels)
        return klist, labels
    except Exception:
        print("[INFO] " + fname + " loading mat5 failed, trying mat7")
    try:
        mat = h5py.File(fname)
        klist = []

        for kr in mat[kernel_table][0]:
            k = np.array(mat[kr], dtype=np.double)
            klist.append(k)

        labels = np.array(mat[label_table], dtype=int)
        labels = labels - np.min(labels)
        return klist, labels[0]
    except Exception as e:
        print("[INFO] loading mat7 failed ", e)

    raise RuntimeError("could not open mat file " + fname)


def load_dataset_from_spec(dataset_spec):
    """Load a dataset into memory from a dataset specification.

    Parameters
    ----------
    dataset_spec: {
       "train": <name-of-hd5-train-file>,
       "test": <name-of-hd5-test-file>,
       "kernels": <name-of-kernel-name-list>,
       "labels": <name-of-target-table>
    }
    """

    if "data" in dataset_spec.keys():
        # one large kernel matrix provided, split into 5 train/test splits
        kernels, targets = load_kernels(
            dataset_spec["data"], dataset_spec["kernels"], dataset_spec["labels"]
        )
        ind = np.arange(kernels[0].shape[0])

        dataset = {}
        dataset["name"] = dataset_spec["name"]
        dataset["test_kernels"] = []
        dataset["train_kernels"] = []
        dataset["cv_train_kernels"] = []
        dataset["test_targets"] = []
        dataset["train_targets"] = []
        dataset["cv_train_targets"] = []

        for i in range(0, dataset_spec["repeat"]):
            train_ind, test_ind = train_test_split(
                ind, test_size=dataset_spec["split"], random_state=i, stratify=targets
            )
            train_target = targets[train_ind]
            test_target = targets[test_ind]
            cv_train_ind = train_ind
            if train_ind.shape[0] > 150:
                _, cv_train_ind = train_test_split(
                    train_ind,
                    test_size=150 / train_ind.shape[0],
                    random_state=0,
                    stratify=train_target,
                )

            cv_train_target = targets[cv_train_ind]
            train_Ks = []
            cv_train_Ks = []
            test_Ks = []
            for K in kernels:
                train_K = K[train_ind, :][:, train_ind]
                cv_train_K = K[cv_train_ind, :][:, cv_train_ind]
                test_K = K[test_ind, :][:, train_ind]
                train_Ks.append(train_K)
                test_Ks.append(test_K)
                cv_train_Ks.append(cv_train_K)
            dataset["test_kernels"].append(test_Ks)
            dataset["train_kernels"].append(train_Ks)
            dataset["cv_train_kernels"].append(cv_train_Ks)
            dataset["test_targets"].append(test_target)
            dataset["train_targets"].append(train_target)
            dataset["cv_train_targets"].append(cv_train_target)

        return dataset
    else:
        # there is only one CV run, as there is one dedicated test dataset
        test_kernels, test_targets = load_kernels(
            dataset_spec["test"], dataset_spec["kernels"], dataset_spec["labels"]
        )
        train_kernels, train_targets = load_kernels(
            dataset_spec["train"], dataset_spec["kernels"], dataset_spec["labels"]
        )

        train_ind = np.arange(train_kernels[0].shape[0])
        cv_train_ind = train_ind
        if train_ind.shape[0] > 150:
            _, cv_train_ind = train_test_split(
                train_ind,
                test_size=150 / train_ind.shape[0],
                random_state=0,
                stratify=train_targets,
            )

        cv_train_target = train_targets[cv_train_ind]
        cv_train_kernels = []
        for K in train_kernels:
            cv_train_K = K[cv_train_ind, :][:, cv_train_ind]
            cv_train_kernels.append(cv_train_K)

        dataset = {}
        dataset["name"] = dataset_spec["name"]
        dataset["test_kernels"] = [test_kernels]
        dataset["train_kernels"] = [train_kernels]
        dataset["cv_train_kernels"] = [cv_train_kernels]
        dataset["test_targets"] = [test_targets]
        dataset["train_targets"] = [train_targets]
        dataset["cv_train_targets"] = [cv_train_target]
        return dataset


def extract_setup(jdict):
    """Extract the experimental setup (datasets + classifiers) from an experiment json file.

    Parameters
    ----------
    jdict, dictionary of the experiment json

    Return
    ------
    list of dataset specifications (load individually on demand with
    load_dataset_from_spec(), list of classifiers
    """
    datasets = jdict["datasets"]
    classifiers = []

    for estimator_spec in jdict["estimators"]:
        import_string = ""
        for import_spec in estimator_spec["imports"]:
            _str = (
                "from "
                + import_spec["from"]
                + " import "
                + ", ".join(import_spec["import"])
            )
            import_string = import_string + _str + "\n"
        exec(import_string, globals())
        clf = eval(estimator_spec["construction"], globals())
        data_constructor = eval(estimator_spec["XConstruction"], globals(), locals())
        parameters = {
            estimator_spec["parameterPrefix"] + k: v
            for k, v in estimator_spec["parameters"].items()
        }

        enforce = {
            estimator_spec["parameterPrefix"] + k: v
            for k, v in estimator_spec["enforce"].items()
        }
        # print(enforce)
        estimator = {}
        estimator["name"] = estimator_spec["name"]
        estimator["estimator"] = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=-1)
        estimator["enforce"] = enforce
        estimator["constructor"] = data_constructor
        classifiers.append(estimator)

    return datasets, classifiers


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("expected one argument: askf_bench <experiment-specification.json>")
        exit(1)
    exp_file = str(sys.argv[1])
    with open(exp_file) as f:
        j = json.load(f)
        outname = j["name"]
        datasets, classifiers = extract_setup(j)
        # train and construct output
        findings = []
        dc = -1
        for dataset_spec in datasets:
            dc = dc + 1
            try:
                print(
                    f"[INFO {datetime.datetime.now()}] loading dataset: (",
                    dataset_spec["train"]
                    if "train" in dataset_spec
                    else dataset_spec["data"],
                    ")",
                )
                dataset = load_dataset_from_spec(dataset_spec)
                name = dataset["name"]
                estimator_results = []
                for clf in classifiers:
                    clf_name = clf["name"]
                    repeats = len(dataset["train_targets"])
                    clf_results = {
                        "name": clf_name,
                        "train_score": [],
                        "test_score": [],
                        "cv_time": [],
                        "cv_results": [],
                        "best_index": [],
                    }
                    print(
                        f"[INFO {datetime.datetime.now()}] fitting classifier: ",
                        clf_name,
                    )
                    for i in range(0, repeats):
                        print(f"[INFO {datetime.datetime.now()}] repeat ", i)
                        Ktrain = clf["constructor"](dataset["train_kernels"][i])
                        Kcvtrain = clf["constructor"](dataset["cv_train_kernels"][i])
                        Ktest = clf["constructor"](dataset["test_kernels"][i])

                        clf["estimator"].fit(Kcvtrain, dataset["cv_train_targets"][i])

                        clf["estimator"].best_estimator_.set_params(**clf["enforce"])

                        start = time.time()
                        clf["estimator"].best_estimator_.fit(
                            Ktrain, dataset["train_targets"][i]
                        )
                        end = time.time()

                        score_test = clf["estimator"].best_estimator_.score(
                            Ktest, dataset["test_targets"][i]
                        )
                        score_train = clf["estimator"].best_estimator_.score(
                            Ktrain, dataset["train_targets"][i]
                        )
                        print(
                            f"[INFO {datetime.datetime.now()}] score train, score test, time taken: ",
                            score_train,
                            " ",
                            score_test,
                            " ",
                            end - start,
                        )
                        clf_results["train_score"].append(score_train)
                        clf_results["test_score"].append(score_test)
                        clf_results["cv_time"].append(end - start)
                        clf_results["cv_results"].append(clf["estimator"].cv_results_)
                        clf_results["best_index"].append(clf["estimator"].best_index_)

                    estimator_results.append(clf_results)
                dataset_findings = {"dataset_name": name, "findings": estimator_results}
                findings.append(dataset_findings)
                local_outname = (
                    outname
                    + "_"
                    + str(dc)
                    + "_of_"
                    + str(len(datasets))
                    + "_experiment_results.json"
                )
                with open(local_outname, "w") as outfile:
                    json.dump(findings, outfile, cls=NumpyEncoder)
            except Exception as e:
                print(
                    f"[ERROR {datetime.datetime.now()}] an error ocurred with dataset: ",
                    dataset_spec["test"],
                    " ",
                    dataset_spec["train"],
                    " ",
                    e,
                )

        with open(outname + "_experiment_results.json", "w") as outfile:
            json.dump(findings, outfile, cls=NumpyEncoder)

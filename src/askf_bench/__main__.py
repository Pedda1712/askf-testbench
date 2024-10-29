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

import sys
import json
import h5py
import scipy.io
import numpy as np
from sklearn.model_selection import GridSearchCV


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
    except Exception:
        print("[INFO] loading mat7 failed")

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
    test_kernels, test_targets = load_kernels(
        dataset_spec["test"], dataset_spec["kernels"], dataset_spec["labels"]
    )
    train_kernels, train_targets = load_kernels(
        dataset_spec["train"], dataset_spec["kernels"], dataset_spec["labels"]
    )
    dataset = {}
    dataset["name"] = dataset_spec["name"]
    dataset["test_kernels"] = test_kernels
    dataset["train_kernels"] = train_kernels
    dataset["test_targets"] = test_targets
    dataset["train_targets"] = train_targets
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
        estimator = {}
        estimator["name"] = estimator_spec["name"]
        estimator["estimator"] = GridSearchCV(estimator=clf, param_grid=parameters)
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
    exp_file = str(sys.argv[1])
    with open(exp_file) as f:
        j = json.load(f)
        outname = j["name"]
        datasets, classifiers = extract_setup(j)
        # train and construct output
        findings = []
        i = -1
        for dataset_spec in datasets:
            i = i + 1
            try:
                print(
                    "[INFO] loading dataset: (",
                    dataset_spec["train"],
                    ", ",
                    dataset_spec["test"],
                    ")",
                )
                dataset = load_dataset_from_spec(dataset_spec)
                name = dataset["name"]
                estimator_results = []
                for clf in classifiers:
                    clf_name = clf["name"]
                    print("[INFO] fitting classifier: ", clf_name)
                    Ktrain = clf["constructor"](dataset["train_kernels"])
                    Ktest = clf["constructor"](dataset["test_kernels"])
                    clf["estimator"].fit(Ktrain, dataset["train_targets"])
                    score_test = clf["estimator"].score(Ktest, dataset["test_targets"])
                    score_train = clf["estimator"].score(
                        Ktrain, dataset["train_targets"]
                    )
                    print("[INFO] score train/test: ", score_train, " ", score_test)
                    clf_results = {
                        "name": clf_name,
                        "train_score": score_train,
                        "test_score": score_test,
                        "cv_results": clf["estimator"].cv_results_,
                        "best_index": clf["estimator"].best_index_,
                    }
                    estimator_results.append(clf_results)
                dataset_findings = {"dataset_name": name, "findings": estimator_results}
                findings.append(dataset_findings)
                local_outname = (
                    outname
                    + "_"
                    + str(i)
                    + "_of_"
                    + str(len(datasets))
                    + "_experiment_results.json"
                )
                with open(local_outname, "w") as outfile:
                    json.dump(findings, outfile, cls=NumpyEncoder)
            except Exception as e:
                print(
                    "[ERROR] an error ocurred with dataset: ",
                    dataset_spec["test"],
                    " ",
                    dataset_spec["train"],
                    " ",
                    e
                )

        with open(outname + "_experiment_results.json", "w") as outfile:
            json.dump(findings, outfile, cls=NumpyEncoder)

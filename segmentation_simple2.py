import faulthandler
import glob
import gzip
import os
import pickle
from time import time
from typing import Mapping, Optional

import experitur
import h5py
import numpy as np
import pandas as pd
import PIL.Image
import sklearn.metrics
import supervisely as sly
import tqdm
from experitur import Experiment, Trial
from experitur.core.configurators import Const
from experitur.core.context import get_current_context
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from timer_cm import Timer

from segmenter import (
    MinIntensityPreSelector,
    MultiscaleBasicFeatures,
    NullFeatures,
    Segmenter,
)

faulthandler.enable()


def files_df(pattern, name):
    df = pd.Series(glob.glob(pattern)).to_frame(name)
    df["id"] = df[name].map(lambda s: os.path.splitext(os.path.basename(s))[0])
    df = df.set_index("id", drop=True)
    return df


def load_dataset(path) -> pd.DataFrame:
    images = files_df(os.path.join(path, "img", "*.*"), "image_fn")
    print(f"{len(images)} images.")

    labels = files_df(os.path.join(path, "masks_machine", "*.*"), "labels_fn")
    print(f"{len(labels)} masks.")

    file_data = images.join(labels, how="outer")

    assert not file_data.isna().any().any(), "images and labels do not match"

    data = pd.read_csv(os.path.join(path, "image_meta.csv"), index_col=0)
    # Remove file name extension
    data.index = data.index.map(lambda s: os.path.splitext(s)[0])
    print(f"{len(data)} metadata.")

    missing = file_data[~file_data.index.isin(data.index)]
    if missing.size:
        print("Entries missing in metadata:")
        print(missing)

    data = data.join(file_data, how="inner")

    print(f"Loaded {len(data)} entries.")

    return data


def save_gz(fn: str, array: np.ndarray):
    with gzip.GzipFile(fn, "w") as f:
        np.save(f, array)


def load_gz(fn: str, **kwargs) -> np.ndarray:
    with gzip.GzipFile(fn, "r") as f:
        return np.load(f, **kwargs)


@Experiment()
def dataset(trial: Trial):

    project = sly.read_single_project(trial["path"])
    foreground_class = trial.get("foreground_class", "Organism")
    background_class = trial.get("background_class", "Light")

    data = {}

    for dset in project.datasets:
        dset: sly.Dataset
        labels_dir = os.path.join(trial.wdir, dset.name, "labels")
        classes_dir = os.path.join(trial.wdir, dset.name, "classes")
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(classes_dir, exist_ok=True)

        for item_name in tqdm.tqdm(dset.get_items_names(), desc=dset.name):
            item_id = os.path.splitext(item_name)[0].replace("  ", " ")

            ann: sly.Annotation = dset.get_ann(item_name, project.meta)

            # Draw region and class images
            label_image = np.zeros(ann.img_size, dtype="int")
            class_image = np.full(ann.img_size, -1, dtype="int")

            for i, label in enumerate(ann.labels):
                if not isinstance(label.geometry, sly.Bitmap):
                    continue

                mask = label.geometry.data
                h, w = mask.shape
                origin = label.geometry.origin
                sl = (
                    slice(origin.row, origin.row + h),
                    slice(origin.col, origin.col + w),
                )

                if label.obj_class.name == foreground_class:
                    label_image[sl] = mask * i
                    class_image[sl][mask] = 1
                elif label.obj_class.name == background_class:
                    class_image[sl][mask] = 0

            labels_fn = os.path.join(labels_dir, item_id + ".np.gz")
            save_gz(labels_fn, label_image)

            classes_fn = os.path.join(classes_dir, item_id + ".np.gz")
            save_gz(classes_fn, label_image)

            data[item_id] = dict(
                image_fn=dset.get_img_path(item_name),
                labels_fn=labels_fn,
                classes_fn=classes_fn,
            )

    data = pd.DataFrame.from_dict(data, orient="index")

    print(f"Loaded {len(data)} items.")

    meta = pd.read_csv(os.path.join(trial["path"], "image_meta.csv"), index_col=0)
    print(f"Loaded {len(meta)} metadata.")

    missing = data[~data.index.isin(meta.index)]
    if missing.size:
        raise ValueError(f"{len(missing)} entries missing in metadata: {missing.index}")

    data = data.join(meta, how="inner")

    # Add split info
    nsplits = trial.get("nsplits", 5)
    groupby = trial.get("groupby", None)
    if groupby is not None:
        split = data.groupby(groupby, group_keys=False).apply(
            lambda group: pd.Series(np.arange(len(group)) % nsplits, index=group.index)
        )
    else:
        split = pd.Series(np.arange(len(data)) % nsplits)

    data["split"] = split

    data.to_csv(os.path.join(trial.wdir, "dataset.csv"))


@Experiment()
def extract_features(trial: Trial):
    dataset_trial = experitur.get_trial(trial["dataset_trial_id"])
    dataset = pd.read_csv(dataset_trial.find_file("dataset.csv"), index_col=0)

    cls = trial.choice("cls", [MultiscaleBasicFeatures, NullFeatures])
    feature_extractor = trial.call(cls)

    # Save feature extractor
    feature_extractor_fn = trial.file("feature_extractor.pkl.gz")
    with gzip.open(feature_extractor_fn, "wb") as f:
        pickle.dump(feature_extractor, f)

    trial.save()

    features_fn = os.path.join(trial.wdir, "features.h5")
    with h5py.File(features_fn, "w") as features_h5:
        with Timer("calculate_features") as t:
            for row in tqdm.tqdm(
                dataset.itertuples(), desc="Features", total=len(dataset)
            ):
                image = np.array(PIL.Image.open(row.image_fn).convert("L"))
                f = feature_extractor(image)

                features_h5.create_dataset(row.Index, data=f, compression="gzip")

        time_calculate_features = float(t.elapsed) / len(dataset)

    return {"time_calculate_features": time_calculate_features}


def _build_trainingset(features_h5, dataset, preselector):
    X = []
    y = []
    for row in tqdm.tqdm(
        dataset.itertuples(), desc="Building training set...", total=len(dataset)
    ):
        image = np.array(PIL.Image.open(row.image_fn).convert("L"))

        # Classes: Each pixel is assigned a class
        # 0=background, 1=foreground, -1=ignore
        classes: np.ndarray = load_gz(row.classes_fn)

        # Load precalculated features
        features: np.ndarray = features_h5[row.Index][:]  # type: ignore

        # Select only annotated pixels
        mask = classes >= 0

        # Apply preselector
        if preselector is not None:
            mask &= preselector(image)

        X.append(features[mask])
        y.append(classes[mask])

    return np.concatenate(X), np.concatenate(y)


def _configure(obj, **kwargs):
    for k, v in kwargs.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


@Experiment()
def train(trial: Trial):
    extract_features_trial = experitur.get_trial(trial["extract_features_trial_id"])
    dataset_trial = experitur.get_trial(extract_features_trial["dataset_trial_id"])
    dataset = pd.read_csv(dataset_trial.find_file("dataset.csv"), index_col=0)
    split = trial.get("split", None)

    if split is None:
        print(f"Training on complete dataset ({len(dataset):,d} samples)")
    else:
        dataset = dataset[dataset["split"] != split]
        print(
            f"Training on training split ({split}) dataset ({len(dataset):,d} samples)"
        )

    preselector_cls = trial.choice("preselector", [None, MinIntensityPreSelector])
    preselector = (
        None
        if preselector_cls is None
        else trial.prefixed("preselector_").call(preselector_cls)
    )

    # Save preselector
    preselector_fn = trial.file("preselector.pkl.gz")
    with gzip.open(preselector_fn, "wb") as f:
        pickle.dump(preselector, f)

    classifier_cls = trial.choice(
        "classifier", [RandomForestClassifier, LinearSVC, DummyClassifier]
    )
    classifier = trial.prefixed("classifier_").call(classifier_cls)

    # Configure classifier for training
    _configure(classifier, n_jobs=-1, verbose=1)

    trial.save()

    # Build training set
    features_fn = extract_features_trial.find_file("features.h5")
    with h5py.File(features_fn, "r") as features_h5:
        X, y = _build_trainingset(features_h5, dataset, preselector)

    assert X.shape[0] == y.shape[0], f"X: {X.shape}, y: {y.shape}"

    preselector_cls = trial.choice("preselector", [None, MinIntensityPreSelector])
    preselector = (
        None
        if preselector_cls is None
        else trial.prefixed("preselector_").call(preselector_cls)
    )

    print("Fitting classifier...")
    with Timer("train") as t_fit:
        classifier.fit(X, y)

    # Save preselector
    classifier_fn = trial.file("classifier.pkl.gz")
    with gzip.open(classifier_fn, "wb") as f:
        pickle.dump(classifier, f)

    print("Evaluating classifier...")
    with Timer("predict") as t_predict:
        y_pred = classifier.predict(X)

    # Evaluate (without post-processing)
    (
        precision,
        recall,
        f_score,
        support,
    ) = sklearn.metrics.precision_recall_fscore_support(y, y_pred, average="binary")

    accuracy = sklearn.metrics.accuracy_score(y, y_pred)

    return {
        "t_fit": float(t_fit.elapsed),
        "t_predict": float(t_predict.elapsed),
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "accuracy": accuracy,
        "support": support,
    }


def _evaluate_segments(labels_true, labels_pred):
    ...


def _evaluate_classes(classes_true, classes_pred):
    ...


@Experiment()
def evaluate_postprocessor(trial: Trial):
    train_trial = experitur.get_trial(trial["train_trial_id"])
    extract_features_trial = experitur.get_trial(
        train_trial["extract_features_trial_id"]
    )
    dataset_trial = experitur.get_trial(extract_features_trial["dataset_trial_id"])
    dataset = pd.read_csv(dataset_trial.find_file("dataset.csv"), index_col=0)

    # Load feature extractor
    feature_extractor_fn = extract_features_trial.find_file("feature_extractor.pkl.gz")
    with gzip.open(feature_extractor_fn, "rb") as f:
        feature_extractor = pickle.load(f)

    # Load preselector
    preselector_fn = train_trial.find_file("preselector.pkl.gz")
    with gzip.open(preselector_fn, "rb") as f:
        preselector = pickle.load(f)

    # Load classifier
    classifier_fn = extract_features_trial.find_file("classifier.pkl.gz")
    with gzip.open(classifier_fn, "rb") as f:
        classifier = pickle.load(f)

    # Initialize postprocessor
    postprocessor_cls = trial.choice("postprocessor", [])
    postprocessor = trial.prefixed("postprocessor_").call(postprocessor_cls)

    segmenter = Segmenter(
        feature_extractor=feature_extractor,
        classifier=classifier,
        preselector=preselector,
        postprocessor=postprocessor,
    )

    split = trial.get("split", None)
    if split is None:
        print(f"Evaluating on complete dataset ({len(dataset):,d} samples)")
    else:
        dataset = dataset[dataset["split"] != split]
        print(
            f"Evaluating on training split ({split}) dataset ({len(dataset):,d} samples)"
        )

    features_fn = extract_features_trial.find_file("features.h5")
    with h5py.File(features_fn, "r") as features_h5:
        for row in tqdm.tqdm(
            dataset.itertuples(), desc="Building validation set...", total=len(dataset)
        ):
            image = np.array(PIL.Image.open(row.image_fn).convert("L"))

            # Classes: Each pixel is assigned a class
            # TODO: Meaning of values?
            classes: np.ndarray = load_gz(row.classes_fn)

            # Each pixel is assigned a segment
            labels: np.ndarray = load_gz(row.labels_fn)

            # Load precalculated features
            features: np.ndarray = features_h5[row.Index][:]  # type: ignore

            # Apply rest of segmentation pipeline
            mask = segmenter.preselect(image)
            mask = segmenter.predict_pixels(features, mask)
            labels_pred = segmenter.postprocess(mask, image)

            # Evaluate
            _evaluate_classes(classes, labels_pred > 0)
            _evaluate_segments(labels, labels_pred)

    raise NotImplementedError()


def run_stage(
    trial: Trial, experiment: Experiment, config: Optional[Mapping] = None, **kwargs
):
    exp = experiment.child(configurator=Const(trial, **kwargs))

    ctx = get_current_context()

    exp.run()

    sub_trial = exp.get_matching_trials().one()

    while not (sub_trial.is_successful or sub_trial.is_failed or sub_trial.is_zombie):
        time.sleep(10)
        sub_trial = ctx.get_trial(sub_trial.id)

    if sub_trial.is_failed or sub_trial.is_zombie:
        raise ValueError(f"Trial {sub_trial.id} was unsuccessful")

    trial.update(sub_trial)
    trial.update_result(sub_trial.result)

    return sub_trial


# max_depth
@Experiment(
    configurator=Const(
        dataset_path="data/22-06-14-LOKI",
        dataset_nsplits=5,
        dataset_groupby="node_id",
        # Feature Extractor
        extract_features_sigma_max=32,
        extract_features_edges=True,
        extract_features_intensity=True,
        extract_features_texture=True,
        # Classifier
        train_classifier_max_depth=10,
    )
)
def train_eval(trial: Trial):

    dataset_trial = run_stage(
        trial.prefixed("dataset_"),
        dataset,
    )

    extract_features_trial = run_stage(
        trial.prefixed("extract_features_"),
        extract_features,
        dataset_trial_id=dataset_trial.id,
    )

    results = []
    for split in range(dataset_trial["nsplits"]):
        train_trial = run_stage(
            trial.prefixed("train_"),
            train,
            split=split,
            extract_features_trial_id=extract_features_trial.id,
        )

        eval_trial = run_stage(
            trial.prefixed("eval_"),
            evaluate_postprocessor,
            train_trial_id=train_trial.id,
            split=split,
        )

        results.append(eval_trial.result)

    results = pd.DataFrame(results)
    results.agg(["mean", "std"])

    results.to_pickle(trial.file("results.pkl"))

    raise NotImplementedError()

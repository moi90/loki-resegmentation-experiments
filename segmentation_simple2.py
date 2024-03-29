import faulthandler
import glob
import gzip
import logging
import os
import pickle
from functools import partial
from typing import List, Mapping, Optional

import experitur
import h5py
import numpy as np
import pandas as pd
import PIL.Image
import scipy.optimize
import shapely.geometry
import skimage.color
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.util
import sklearn.metrics
import supervisely as sly
import tqdm
from envparse import env
from experitur import Experiment, Trial
from experitur.configurators import AdditiveConfiguratorChain, RandomGrid, SKOpt
from experitur.core.configurators import Const, Reset
from experitur.core.context import get_current_context
from experitur.core.experiment import SkipTrial
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from util import load_gz, save_gz
from timer_cm import Timer
from sklearn.linear_model import LogisticRegression

from _version import get_versions
from segmenter import (
    DefaultPostProcessor,
    MinIntensityPreSelector,
    MultiscaleBasicFeatures,
    NullFeatures,
    Segmenter,
    WatershedPostProcessor,
)

faulthandler.enable()

logging.getLogger("imageio").setLevel(logging.ERROR)

N_JOBS = env("N_JOBS", cast=int, default=1)

meta = {"version": get_versions()["version"]}


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


@Experiment(meta=meta)
def dataset(trial: Trial):

    project = sly.read_single_project(trial["path"])
    foreground_class = trial.get("foreground_class", "Foreground")
    background_class = trial.get("background_class", "Background")
    relabel = trial.get("relabel", False)

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
                    label_image[sl] = mask * (i + 1)
                    class_image[sl][mask] = 1
                elif label.obj_class.name == background_class:
                    class_image[sl][mask] = 0

            if relabel:
                # Relabel the image
                # This merges connected regions and splits disconnected regions
                label_image = skimage.measure.label(class_image)

            assert (class_image == 1).any(), f"{item_name} has no foreground pixels"
            assert (label_image > 0).any(), f"{item_name} has no labeled regions"

            labels_fn = os.path.join(labels_dir, item_id + ".np.gz")
            save_gz(labels_fn, label_image)

            classes_fn = os.path.join(classes_dir, item_id + ".np.gz")
            save_gz(classes_fn, class_image)

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


def gz_dump(filename, obj):
    with gzip.open(filename, "wb") as f:
        s = pickle.dumps(obj)
        f.write(s)
        return len(s)


@Experiment(meta=meta)
def extract_features(trial: Trial):
    dataset_trial = experitur.get_trial(trial["dataset_trial_id"])
    dataset = pd.read_csv(dataset_trial.find_file("dataset.csv"), index_col=0)

    cls = trial.choice("cls", [MultiscaleBasicFeatures, NullFeatures])
    feature_extractor = trial.prefixed(f"{cls.__name__}_").call(cls)

    # Save feature extractor
    gz_dump(trial.file("feature_extractor.pkl.gz"), feature_extractor)

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


def _augment_classes(classes, bg_margin=0, bg_width=0):
    """
    Set pixels to background (0) around any foreground objects (1).
    """
    foreground = classes == 1
    if bg_margin:
        margin = skimage.morphology.binary_dilation(
            foreground, skimage.morphology.disk(bg_margin)
        )
    else:
        margin = foreground

    background = (
        skimage.morphology.binary_dilation(margin, skimage.morphology.disk(bg_width))
        & ~margin
    )

    classes[background] = 0

    return classes


def _build_trainingset(
    features_h5, dataset, preselector, bg_margin=0, bg_width=0, equal_weight=False
):
    X = []
    y = []

    if equal_weight:
        sample_weight = []
    else:
        sample_weight = None

    for row in tqdm.tqdm(
        dataset.itertuples(), desc="Building training set...", total=len(dataset)
    ):
        image = np.array(PIL.Image.open(row.image_fn).convert("L"))

        # Classes: Each pixel is assigned a class
        # 0=background, 1=foreground, -1=ignore
        classes: np.ndarray = load_gz(row.classes_fn)

        # Load precalculated features
        features: np.ndarray = features_h5[row.Index][:]  # type: ignore

        # TODO: Grow annotated area around objects
        classes = _augment_classes(classes, bg_margin, bg_width)

        # Select only annotated pixels
        mask = classes >= 0

        # Apply preselector
        if preselector is not None:
            mask &= preselector(image)

        X.append(features[mask])
        y.append(classes[mask])

        if sample_weight is not None:
            n_samples = mask.sum()
            sample_weight.append(np.full(n_samples, 1 / n_samples))

    if sample_weight is not None:
        sample_weight = np.concatenate(sample_weight)

    return (np.concatenate(X), np.concatenate(y), sample_weight)


def _configure(obj, **kwargs):
    for k, v in kwargs.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


@Experiment(meta=meta, maximize=["fgAP"])
def train(trial: Trial):
    extract_features_trial = experitur.get_trial(trial["extract_features_trial_id"])
    dataset_trial = experitur.get_trial(extract_features_trial["dataset_trial_id"])
    dataset = pd.read_csv(dataset_trial.find_file("dataset.csv"), index_col=0)

    split = trial.get("split", None)
    if split is None:
        print("Training on complete dataset")
    else:
        dataset = dataset[dataset["split"] != split]
        print(f"Training on training split {split}")

    fast_n = trial.get("fast_n", None)
    if fast_n is not None:
        dataset = dataset.iloc[:fast_n]

    print(f"{len(dataset)} samples")

    preselector_cls = trial.choice("preselector", [None, MinIntensityPreSelector])
    preselector = (
        None
        if preselector_cls is None
        else trial.prefixed(f"preselector_{preselector_cls.__name__}_").call(
            preselector_cls
        )
    )

    # Save preselector
    gz_dump(trial.file("preselector.pkl.gz"), preselector)

    classifier_cls = trial.choice(
        "classifier",
        [RandomForestClassifier, LinearSVC, DummyClassifier, LogisticRegression],
    )
    classifier = trial.prefixed(f"classifier_{classifier_cls.__name__}_").call(
        classifier_cls
    )

    # Configure classifier for training
    _configure(classifier, n_jobs=N_JOBS, verbose=1)

    trial.save()

    # Build training set
    features_fn = extract_features_trial.find_file("features.h5")
    with h5py.File(features_fn, "r") as features_h5:
        X, y, sample_weight = trial.prefixed("tset_").call(
            partial(_build_trainingset, features_h5, dataset, preselector)
        )

    # Check that y consists only of back- and foreground
    np.testing.assert_array_equal(np.unique(y), np.array([0, 1]))

    trial.save()

    assert X.shape[0] == y.shape[0], f"X: {X.shape}, y: {y.shape}"

    print("Fitting classifier...")
    with Timer("train") as t_fit:
        classifier.fit(X, y, sample_weight=sample_weight)

    # Save classifier
    classifier_size = gz_dump(trial.file("classifier.pkl.gz"), classifier)

    # Set to single-threaded for prediction performance evaluation
    _configure(classifier, n_jobs=1)

    print("Evaluating classifier...")
    with Timer("predict") as t_predict:
        y_pred = classifier.predict(X)

    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X)[:, 1]
        fgAP = sklearn.metrics.average_precision_score(y == 1, probs)
        trial.update_result(fgAP=fgAP)

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
        "classifier_size": classifier_size,
    }


def intersection_over_union(y_true: np.ndarray, y_pred: np.ndarray):
    # Intersection-over-union of two binary classifications
    assert y_true.dtype == np.dtype("bool")
    assert y_pred.dtype == np.dtype("bool")

    return (y_true & y_pred).sum() / (y_true | y_pred).sum()


def bbox_iou(bbox1, bbox2):
    # min_row, min_col, max_row, max_col

    bbox1 = shapely.geometry.box(*bbox1)
    bbox2 = shapely.geometry.box(*bbox2)

    return bbox1.intersection(bbox2).area / bbox1.union(bbox2).area


def _evaluate_segments(labels_true, labels_pred):
    # Clustering metrics
    common_foreground = (labels_true > 0) & (labels_pred > 0)
    (
        homogeneity,
        completeness,
        v_score,
    ) = sklearn.metrics.homogeneity_completeness_v_measure(
        labels_true[common_foreground], labels_pred[common_foreground]
    )

    # Region-based metrics
    regions_true: List[
        skimage.measure._regionprops.RegionProperties
    ] = skimage.measure.regionprops(labels_true)
    regions_pred: List[
        skimage.measure._regionprops.RegionProperties
    ] = skimage.measure.regionprops(labels_pred)

    # Match labels (excluding background)
    # (Ensure that every true segment receives a match)
    n_true = len(regions_true)
    n_pred = len(regions_pred)

    assert n_true > 0, "No foreground regions"

    ious = np.zeros((max(n_true, n_pred), max(n_true, n_pred)))
    for i, r_true in enumerate(regions_true):
        for j, r_pred in enumerate(regions_pred):
            ious[i, j] = intersection_over_union(
                labels_true == r_true.label, labels_pred == r_pred.label
            )

    # Match best regions
    ii, jj = scipy.optimize.linear_sum_assignment(ious, maximize=True)
    assert len(ii) == max(n_true, n_pred)

    # Mean IoU
    # (Requires a match for every true segment)
    mean_iou = ious[ii, jj].mean()

    # Calculate BBox IoU
    mean_bbox_iou = np.mean(
        [
            bbox_iou(regions_true[i].bbox, regions_pred[j].bbox)
            if i < n_true and j < n_pred
            else 0
            for i, j in zip(ii, jj)
        ]
    )

    # Precision / Recall
    iou_true = np.zeros(max(n_true, n_pred))
    iou_true[ii] = ious[ii, jj]
    iou_pred = np.zeros(max(n_true, n_pred))
    iou_pred[jj] = ious[ii, jj]

    precision = (iou_pred > 0.5).sum() / n_pred
    recall = (iou_true > 0.5).sum() / n_true
    _denom = precision + recall
    f1_score = 2 * precision * recall / (_denom if _denom > 0 else 1)

    return {
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_score": v_score,
        "mean_iou": mean_iou,
        "mean_bbox_iou": mean_bbox_iou,
        "label_precision": precision,
        "label_recall": recall,
        "label_f1_score": f1_score,
    }


def _evaluate_classes(classes_true, classes_pred):
    mask = classes_true > -1

    # Pixel-level evaluation
    y_true = classes_true[mask].astype(bool)
    y_pred = classes_pred[mask].astype(bool)

    # Evaluate foreground
    (
        fg_precision,
        fg_recall,
        fg_f1,
        fg_support,
    ) = sklearn.metrics.precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
    )

    # Evaluate background
    (
        bg_precision,
        bg_recall,
        bg_f1,
        bg_support,
    ) = sklearn.metrics.precision_recall_fscore_support(
        ~y_true,
        ~y_pred,
        average="binary",
        zero_division=1,
    )

    mean_precision = (fg_precision + bg_precision) / 2
    mean_recall = (fg_recall + bg_recall) / 2

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

    # Evaluate foreground overlap
    fg_true = classes_true == 1
    fg_pred = classes_pred == 1

    iou = intersection_over_union(fg_true, fg_pred)

    return {
        "px_fg_precision": fg_precision,
        "px_fg_recall": fg_recall,
        "px_fg_f1_score": fg_f1,
        "px_fg_support": fg_support,
        "px_bg_precision": bg_precision,
        "px_bg_recall": bg_recall,
        "px_bg_f1_score": bg_f1,
        "px_bg_support": bg_support,
        "px_mean_precision": mean_precision,
        "px_mean_recall": mean_recall,
        "px_accuracy": accuracy,
        "px_iou": iou,
    }


def draw_segmentation(image: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.dtype == np.dtype(
        "bool"
    ), f"Unexpected dtype for y_true: {y_true.dtype}"
    assert y_pred.dtype == np.dtype(
        "bool"
    ), f"Unexpected dtype for y_pred: {y_pred.dtype}"

    # Assemble label image
    label_image = np.zeros(image.shape[:2], dtype="uint8")
    label_image[y_true & y_pred] = 1  # TP
    label_image[y_true & (~y_pred)] = 2  # FN
    label_image[(~y_true) & y_pred] = 3  # FP

    result = skimage.color.label2rgb(
        label_image, image, colors=["cyan", "magenta", "yellow"], alpha=0.5
    )

    # TODO: Convert result back to image.dtype
    # to avoid Lossy conversion warning

    return result


def summarize(df: pd.DataFrame, percentiles=[0.1, 0.5, 0.9]):
    """
    Return a dictionary of summarized values.

    Args:
        df (pd.DataFrame): Dataframe of shape (n_samples, n_scores).

    Returns:
        Dictionary of <score>_<min|mean|max|10%|...>: <value>
    """
    summary = (
        df.describe(percentiles=percentiles, include=[np.number])
        .drop(index=["count"])
        .unstack()
    )
    summary.index = summary.index.map("_".join)
    return summary.to_dict()


@Experiment(meta=meta)
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
    classifier_fn = train_trial.find_file("classifier.pkl.gz")
    with gzip.open(classifier_fn, "rb") as f:
        classifier = pickle.load(f)

    _configure(classifier, n_jobs=1, verbose=0)

    # Initialize postprocessor
    postprocessor_cls = trial.choice(
        "postprocessor", [DefaultPostProcessor, WatershedPostProcessor]
    )
    postprocessor = trial.prefixed(f"postprocessor_{postprocessor_cls.__name__}_").call(
        postprocessor_cls
    )

    segmenter = Segmenter(
        feature_extractor=feature_extractor,
        classifier=classifier,
        preselector=preselector,
        postprocessor=postprocessor,
    )

    # Save segmenter
    gz_dump(trial.file("segmenter.pkl.gz"), segmenter)

    split = trial.get("split", None)
    if split is None:
        print("Evaluating on complete dataset")
    else:
        dataset = dataset[dataset["split"] == split]
        print(f"Evaluating on validation split {split}")

    fast_n = trial.get("fast_n", None)
    if fast_n is not None:
        dataset = dataset.iloc[:fast_n]

    print(f"{len(dataset)} samples")

    with Timer("eval_val") as t:
        results = []
        with h5py.File(
            extract_features_trial.find_file("features.h5"), "r"
        ) as features_h5:
            for row in tqdm.tqdm(
                dataset.itertuples(), desc="Validating...", total=len(dataset)
            ):
                image = np.array(PIL.Image.open(row.image_fn).convert("L"))

                # Classes: Each pixel is assigned a class
                # 0=background, 1=foreground, -1=ignore
                classes: np.ndarray = load_gz(row.classes_fn)

                # Each pixel is assigned a segment
                labels: np.ndarray = load_gz(row.labels_fn)

                # Load precalculated features
                features: np.ndarray = features_h5[row.Index][:]  # type: ignore

                # Apply rest of segmentation pipeline
                mask = segmenter.preselect(image)
                scores = segmenter.predict_pixels(features, mask)
                labels_pred = segmenter.postprocess(scores, image)

                # Draw
                if trial.get("draw_segmentation", False):
                    img = draw_segmentation(image, classes == 1, labels_pred > 0)
                    skimage.io.imsave(
                        trial.file(f"segmentation/{row.Index}.png", True),
                        img,
                        check_contrast=False,
                    )

                # Evaluate
                scores_classes = _evaluate_classes(
                    classes, (labels_pred > 0).astype(int)
                )
                scores_labels = _evaluate_segments(labels, labels_pred)

                results.append({"id": row.Index, **scores_classes, **scores_labels})

    time_eval = float(t.elapsed) / len(dataset)

    results = pd.DataFrame(results).set_index("id", drop=True)
    results.to_csv(trial.file("scores.csv"))
    summary = summarize(results, percentiles=[0.05, 0.1, 0.5, 0.9])

    return {
        "t_eval": time_eval,
        **results.mean(axis=0, numeric_only=True).to_dict(),
        **summary,
    }


def run_stage(
    trial: Trial, experiment: Experiment, config: Optional[Mapping] = None, **kwargs
):
    exp = experiment.child(configurator=Const(trial, **kwargs))

    exp.run()

    sub_trial = exp.get_matching_trials().one()

    while not (sub_trial.is_successful or sub_trial.is_failed or sub_trial.is_zombie):
        raise SkipTrial("Already running")

    if sub_trial.is_failed or sub_trial.is_zombie:
        raise ValueError(f"Trial {sub_trial.id} was unsuccessful")

    trial.update(sub_trial)
    trial.update_result(sub_trial.result)

    return sub_trial


# prepare_train_min_intensity=24,
@Experiment(
    configurator=Const(
        dataset_path="data/22-11-02-LOKI-raw",
        dataset_nsplits=5,
        dataset_groupby="node_id",
        # Debug
        fast_n=None,
        max_splits=None,
        eval_draw_segmentation=False,
        # Feature Extractor
        extract_features_MultiscaleBasicFeatures_sigma_max=16,
        extract_features_MultiscaleBasicFeatures_edges=True,
        extract_features_MultiscaleBasicFeatures_intensity=True,
        extract_features_MultiscaleBasicFeatures_texture=True,
        # Training set preparation
        train_tset_bg_margin=5,
        train_tset_bg_width=15,
        train_tset_equal_weight=False,
        # Preselector
        train_preselector="MinIntensityPreSelector",
        train_preselector_MinIntensityPreSelector_min_intensity=24,
        train_preselector_MinIntensityPreSelector_dilate=20,
        # Classifier
        train_classifier="RandomForestClassifier",
        train_classifier_RandomForestClassifier_max_depth=5,
        train_classifier_RandomForestClassifier_n_estimators=10,
    ),
    meta=meta,
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

    n_splits = dataset_trial["nsplits"]
    max_splits = trial.get("max_splits", None)
    if max_splits is not None:
        n_splits = min(n_splits, max_splits)

    results = []
    for split in range(n_splits):
        train_trial = run_stage(
            trial.prefixed("train_"),
            train,
            split=split,
            extract_features_trial_id=extract_features_trial.id,
            fast_n=trial.get("fast_n"),
        )

        eval_trial = run_stage(
            trial.prefixed("eval_"),
            evaluate_postprocessor,
            train_trial_id=train_trial.id,
            split=split,
            fast_n=trial.get("fast_n"),
        )

        results.append(eval_trial.result)

    results = pd.DataFrame(results)
    results.to_csv(trial.file("results.csv"))

    # TODO: Sum up times

    return results.mean().to_dict()


def _explore_options(options: Mapping, objective):
    configurators = []

    # Explore individually
    for k, v in options.items():
        configurators.append(RandomGrid({k: v}))

    n_total = sum(len(v) for v in options.values())

    # Optimize
    configurators.append(
        SKOpt(
            {k: SKOpt.Categorical(v) for k, v in options.items()},
            objective=objective,
            n_iter=n_total,
        )
    )

    return AdditiveConfiguratorChain(*configurators, shuffle=True)


train_eval_optimize = Experiment(
    parent=train_eval,
    configurator=[
        # Only evaluate the first split
        Const(max_splits=1),
        AdditiveConfiguratorChain(
            _explore_options(
                {
                    # Preselector
                    "train_preselector_MinIntensityPreSelector_min_intensity": [
                        6,
                        12,
                        24,
                    ],
                    "train_preselector_MinIntensityPreSelector_dilate": [10, 20, 40],
                    # TSet Builder
                    "train_tset_bg_margin": [3, 5, 10],
                    "train_tset_bg_width": [7, 15, 30],
                    "train_tset_equal_weight": [True, False],
                    # Feature Extractor
                    "extract_features_MultiscaleBasicFeatures_edges": [True, False],
                    "extract_features_MultiscaleBasicFeatures_intensity": [True, False],
                    "extract_features_MultiscaleBasicFeatures_sigma_max": [16, 32, 48],
                    "extract_features_MultiscaleBasicFeatures_texture": [True, False],
                    # Classifier
                    "train_classifier_RandomForestClassifier_max_depth": [
                        5,
                        10,
                        20,
                    ],
                    "train_classifier_RandomForestClassifier_n_estimators": [
                        5,
                        10,
                        20,
                    ],
                },
                objective="mean_bbox_iou",
            ),
            # Alternative Preselector
            # Reset("train_preselector*") * Const(train_preselector="None"),
            # Alternative Classifier
            Reset("train_classifier*") * Const(train_classifier="LinearSVC"),
        )
        # SKOpt(
        #     {
        #         "eval_postprocessor_threshold": SKOpt.Categorical([0.5, 0.75, 0.8]),
        #         "eval_postprocessor_smoothing": SKOpt.Categorical([0, 5, 10]),
        #     },
        #     # objective="px_iou",
        #     # objective="px_bg_recall",  # Recognize background (light)
        #     # objective="px_mean_recall",  # Optimize recall of foreground and background
        #     objective="mean_iou",  # Optimize match of labeled regions
        #     n_iter=10,
        # ),
    ],
    maximize=["px_iou", "px_bg_recall", "px_mean_recall", "mean_iou"],
)


train_eval_optimize_watershed = Experiment(
    parent=train_eval,
    configurator=[
        # Only evaluate the first split
        Const(max_splits=1),
        Const(
            # Preselector
            train_preselector="MinIntensityPreSelector",
            train_preselector_MinIntensityPreSelector_dilate=20,
            train_preselector_MinIntensityPreSelector_min_intensity=24,
            # Training Set
            train_tset_bg_margin=5,
            train_tset_bg_width=15,
            train_tset_equal_weight=False,
            # Classifier
            train_classifier_RandomForestClassifier_max_depth=5,
            train_classifier_RandomForestClassifier_n_estimators=10,
            # Feature Extractor
            extract_features_MultiscaleBasicFeatures_edges=True,
            extract_features_MultiscaleBasicFeatures_intensity=True,
            extract_features_MultiscaleBasicFeatures_num_sigma=None,
            extract_features_MultiscaleBasicFeatures_sigma_max=16,
            extract_features_MultiscaleBasicFeatures_sigma_min=0.5,
            extract_features_MultiscaleBasicFeatures_texture=True,
            extract_features_cls="MultiscaleBasicFeatures",
            # Postprocessing
            eval_postprocessor="WatershedPostProcessor",
            eval_postprocessor_WatershedPostProcessor_clear_background=False,
            eval_postprocessor_WatershedPostProcessor_closing=25,
            eval_postprocessor_WatershedPostProcessor_dilate_edges=5,
            eval_postprocessor_WatershedPostProcessor_edges="scores",
            eval_postprocessor_WatershedPostProcessor_min_intensity=64,
            eval_postprocessor_WatershedPostProcessor_min_size=256,  # Empirically determined from dataset (1% percentile)
            eval_postprocessor_WatershedPostProcessor_open_background=0,
            eval_postprocessor_WatershedPostProcessor_q_high=0.99,
            eval_postprocessor_WatershedPostProcessor_q_low=0.5,
            eval_postprocessor_WatershedPostProcessor_relative_closing=0.0,
            eval_postprocessor_WatershedPostProcessor_score_sigma=5,
            eval_postprocessor_WatershedPostProcessor_thr_high=0.85,
            eval_postprocessor_WatershedPostProcessor_thr_low=0.75,
        ),
        (
            Const()
            + (
                Reset("train_classifier*")
                * Const(train_classifier="LogisticRegression")
            )
        ),
        _explore_options(
            {
                # Preselector
                "train_preselector_MinIntensityPreSelector_min_intensity": [
                    6,
                    12,
                    24,
                ],
                "train_preselector_MinIntensityPreSelector_dilate": [10, 20, 40],
                # TSet Builder
                "train_tset_bg_margin": [3, 5, 10],
                "train_tset_bg_width": [7, 15, 30],
                "train_tset_equal_weight": [True, False],
                # Feature Extractor
                "extract_features_MultiscaleBasicFeatures_edges": [True, False],
                "extract_features_MultiscaleBasicFeatures_intensity": [True, False],
                "extract_features_MultiscaleBasicFeatures_sigma_max": [16, 32, 48],
                "extract_features_MultiscaleBasicFeatures_texture": [True, False],
                # Classifier
                "train_classifier_RandomForestClassifier_max_depth": [
                    5,
                    10,
                    20,
                    30,
                ],
                "train_classifier_RandomForestClassifier_n_estimators": [
                    5,
                    10,
                    20,
                ],
                # Postprocessor
                "eval_postprocessor_WatershedPostProcessor_edges": ["image", "scores"],
                # "eval_postprocessor_WatershedPostProcessor_min_size": [
                #     0,
                #     64,
                #     128,
                #     256,
                # ],
                "eval_postprocessor_WatershedPostProcessor_thr_low": [
                    0.125,
                    0.25,
                    0.35,
                    0.5,
                    0.75,
                ],
                "eval_postprocessor_WatershedPostProcessor_q_low": [0.25, 0.5, 0.75],
                "eval_postprocessor_WatershedPostProcessor_thr_high": [
                    0.85,
                    0.90,
                    0.95,
                    0.99,
                    1.0,
                ],
                "eval_postprocessor_WatershedPostProcessor_q_high": [
                    0.95,
                    0.97,
                    0.98,
                    0.99,
                ],
                "eval_postprocessor_WatershedPostProcessor_min_intensity": [
                    0,
                    32,
                    64,
                    128,
                ],
                "eval_postprocessor_WatershedPostProcessor_dilate_edges": [0, 3, 5, 9],
                "eval_postprocessor_WatershedPostProcessor_closing": [
                    0,
                    10,
                    25,
                    #     50,
                    #     75,
                ],
                "eval_postprocessor_WatershedPostProcessor_relative_closing": [
                    0,
                    0.5,
                    1,
                    1.5,
                ],
                "eval_postprocessor_WatershedPostProcessor_open_background": [0, 5, 10],
                "eval_postprocessor_WatershedPostProcessor_score_sigma": [0, 5, 10],
                "eval_postprocessor_WatershedPostProcessor_clear_background": [
                    True,
                    False,
                ],
            },
            # objective="px_iou",
            # objective="px_bg_recall",  # Recognize background (light)
            # objective="px_mean_recall",  # Optimize recall of foreground and background
            objective="mean_bbox_iou",  # Optimize match of labeled regions
        ),
    ],
    maximize=["px_iou", "px_bg_recall", "px_mean_recall", "mean_iou", "mean_bbox_iou"],
)

train_eval_optimize_default_postprocessor = Experiment(
    parent=train_eval,
    configurator=[
        # Only evaluate the first split
        Const(max_splits=1),
        Const(
            # Feature Extractor
            extract_features_MultiscaleBasicFeatures_edges=True,
            extract_features_MultiscaleBasicFeatures_intensity=True,
            extract_features_MultiscaleBasicFeatures_sigma_max=16,
            extract_features_MultiscaleBasicFeatures_sigma_min=0.5,
            extract_features_MultiscaleBasicFeatures_texture=True,
            extract_features_cls="MultiscaleBasicFeatures",
            # Preselector
            train_preselector="MinIntensityPreSelector",
            train_preselector_MinIntensityPreSelector_dilate=20,
            train_preselector_MinIntensityPreSelector_min_intensity=24,
            # Training Set
            train_tset_bg_margin=5,
            train_tset_bg_width=15,
            train_tset_equal_weight=False,
            # Classifier
            train_classifier="RandomForestClassifier",
            train_classifier_RandomForestClassifier_max_depth=20,
            train_classifier_RandomForestClassifier_n_estimators=10,
            # Post-Processor
            eval_postprocessor="DefaultPostProcessor",
            eval_postprocessor_DefaultPostProcessor_closing=10,
            eval_postprocessor_DefaultPostProcessor_min_intensity=64,
            # Empirically determined from dataset (1% percentile)
            eval_postprocessor_DefaultPostProcessor_min_size=256,
            eval_postprocessor_DefaultPostProcessor_relative_closing=0,
            eval_postprocessor_DefaultPostProcessor_smoothing=5,
            eval_postprocessor_DefaultPostProcessor_threshold=0.5,
        ),
        (
            Const()
            + (
                Reset("train_classifier*")
                * Const(train_classifier="LogisticRegression")
            )
        ),
        _explore_options(
            {
                # Preselector
                "train_preselector_MinIntensityPreSelector_min_intensity": [
                    6,
                    12,
                    24,
                ],
                "train_preselector_MinIntensityPreSelector_dilate": [10, 20, 40],
                # TSet Builder
                "train_tset_bg_margin": [3, 5, 10],
                "train_tset_bg_width": [7, 15, 30],
                "train_tset_equal_weight": [True, False],
                # Feature Extractor
                "extract_features_MultiscaleBasicFeatures_edges": [True, False],
                "extract_features_MultiscaleBasicFeatures_intensity": [True, False],
                "extract_features_MultiscaleBasicFeatures_sigma_max": [16, 32, 48],
                "extract_features_MultiscaleBasicFeatures_texture": [True, False],
                # # Classifier
                # "train_classifier_RandomForestClassifier_max_depth": [
                #     5,
                #     10,
                #     20,
                #     30,
                # ],
                # "train_classifier_RandomForestClassifier_n_estimators": [
                #     5,
                #     10,
                #     20,
                # ],
                # Post Processor
                "eval_postprocessor_DefaultPostProcessor_threshold": [
                    0.5,
                    0.8,
                    0.9,
                    0.99,
                ],
                "eval_postprocessor_DefaultPostProcessor_smoothing": [0, 5, 10],
                "eval_postprocessor_DefaultPostProcessor_closing": [
                    0,
                    10,
                    25,
                    50,
                    75,
                ],
                "eval_postprocessor_DefaultPostProcessor_min_intensity": [
                    0,
                    32,
                    64,
                    128,
                ],
                # "eval_postprocessor_DefaultPostProcessor_min_size": [
                #     0,
                #     64,
                #     128,
                #     256,
                # ],
            },
            # objective="px_iou",
            # objective="px_bg_recall",  # Recognize background (light)
            # objective="px_mean_recall",  # Optimize recall of foreground and background
            objective="mean_bbox_iou",  # Optimize match of labeled regions
        ),
    ],
    maximize=["px_iou", "px_bg_recall", "px_mean_recall", "mean_iou", "mean_bbox_iou"],
)


@Experiment(
    configurator=Const(
        dataset_path="data/22-11-02-LOKI-raw",
        eval_postprocessor="WatershedPostProcessor",
        eval_postprocessor_WatershedPostProcessor_closing=25,  # Larger values reduce minimum performance
    ),
    meta=meta,
)
def train_export_full(trial: Trial):

    train_eval_trial = (
        (
            get_current_context()
            .get_trials(
                func="segmentation_simple2.train_eval",
                parameters=trial.todict(),
            )
            .filter(lambda trial: trial.is_successful)
        )
        .best_n(1, maximize="mean_bbox_iou")
        .one()
    )

    dataset_trial = run_stage(
        train_eval_trial.prefixed("dataset_"),
        dataset,
    )

    extract_features_trial = run_stage(
        train_eval_trial.prefixed("extract_features_"),
        extract_features,
        dataset_trial_id=dataset_trial.id,
    )

    evaluate_postprocessor_trial = (
        get_current_context()
        .get_trials(
            "segmentation_simple2.evaluate_postprocessor",
            parameters=train_eval_trial.prefixed("eval_"),
        )
        .one()
    )

    segmenter_fn = evaluate_postprocessor_trial.find_file("segmenter.pkl.gz")
    with gzip.open(segmenter_fn, "rb") as f:
        segmenter: Segmenter = pickle.load(f)

    print("Segmenter:")
    print(" Feature Extractor:", segmenter.feature_extractor)
    print(" Preselector:", segmenter.preselector)
    print(" Classifier:", segmenter.classifier)
    print(" Postprocessor:", segmenter.postprocessor)

    # Train on full dataset
    train_trial = run_stage(
        train_eval_trial.prefixed("train_"),
        train,
        split=None,
        extract_features_trial_id=extract_features_trial.id,
        fast_n=trial.get("fast_n"),
    )

    # Load classifier
    classifier_fn = train_trial.find_file("classifier.pkl.gz")
    with gzip.open(classifier_fn, "rb") as f:
        classifier = pickle.load(f)

    # Update classifier
    segmenter.classifier = classifier

    # Save segmenter
    segmenter_fn = trial.file("segmenter.pkl.gz")
    gz_dump(segmenter_fn, segmenter)

    print("Segmenter saved as:", segmenter_fn)

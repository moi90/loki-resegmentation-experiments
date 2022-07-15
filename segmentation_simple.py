from functools import partial
import hashlib
import pickle
from typing import Any, Tuple, overload
from experitur import Trial, Experiment
from experitur.configurators import Const
import os
import glob
import PIL.Image
from experitur.configurators import Grid
from experitur.core.configurators import (
    AdditiveConfiguratorChain,
    FilterConfig,
    RandomGrid,
)
import numpy as np
import skimage.feature
import abc
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import tqdm
import sklearn.metrics
import skimage.measure
import shapely.geometry
import h5py
import json
from skimage.future import predict_segmenter
from sklearn.dummy import DummyClassifier
from timer_cm import Timer
import skimage.morphology

import faulthandler

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


def __iter__(self):
    image_meta = self._load_image_meta()

    for image_fn, labels_fn in self._find_entries():
        image = np.array(PIL.Image.open(image_fn).convert("L"))
        labels = np.array(PIL.Image.open(labels_fn).convert("L"))

        yield (image, labels, image_meta.loc[os.path.basename(image_fn)].todict())


class FeatureExtractor(abc.ABC):
    @abc.abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class MultiscaleBasicFeatures(FeatureExtractor):
    def __init__(
        self,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=0.5,
        sigma_max=16,
        num_sigma=None,
    ):
        self.intensity = intensity
        self.edges = edges
        self.texture = texture
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_sigma = num_sigma

    def __call__(self, image):
        return skimage.feature.multiscale_basic_features(
            image,
            intensity=self.intensity,
            edges=self.edges,
            texture=self.texture,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            num_sigma=self.num_sigma,
            num_workers=1,
        )


class NullFeatures(FeatureExtractor):
    def __call__(self, image):
        # Ensure HxWxC
        return image.reshape(image.shape[:2] + (-1,))


class H5Cache:
    def __init__(self, location, cleanup=False):
        self.location = location
        self.cleanup = True
        self._h5f = None

    def open(self):
        assert self._h5f is None
        self._h5f = h5py.File(self.location, "a")
        return self

    def close(self):
        assert self._h5f is not None
        try:
            self._h5f.close()
        except:
            pass

        self._h5f = None

        if self.cleanup:
            try:
                os.remove(self.location)
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    def __getitem__(self, key) -> np.ndarray:
        assert self._h5f is not None

        return self._h5f[key][...]

    def __setitem__(self, key, value):
        assert self._h5f is not None

        self._h5f.create_dataset(key, data=value, compression="gzip")

    def __contains__(self, key):
        assert self._h5f is not None

        return key in self._h5f


def calculate_features(
    dataset: pd.DataFrame, cache: H5Cache, feature_extractor: FeatureExtractor
):
    for row in tqdm.tqdm(dataset.itertuples(), desc="Features", total=len(dataset)):
        key = row.Index + "/features"

        if key not in cache:
            image = np.array(PIL.Image.open(row.image_fn).convert("L"))
            cache[key] = feature_extractor(image)


def _augment_labels(labels, bg_margin=0, bg_width=0):
    foreground = labels == 1
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

    labels[background] = 2

    return labels


def prepare_training_set(
    dataset: pd.DataFrame,
    cache: H5Cache,
    *,
    bg_margin=0,
    bg_width=0,
    min_intensity=0,
):
    def _calc():
        for row in tqdm.tqdm(dataset.itertuples(), desc="Prepare", total=len(dataset)):
            labels = np.array(PIL.Image.open(row.labels_fn).convert("L"))

            if bg_width > 0:
                labels = _augment_labels(labels, bg_margin, bg_width)

            if min_intensity > 0:
                # Reset foreground to 0 where image < min_intensity
                image = np.array(PIL.Image.open(row.image_fn).convert("L"))
                labels[labels == 1 & (image < min_intensity)] = 0

            features = cache[row.Index + "/features"]
            mask = labels > 0
            yield features[mask].copy(), labels[mask].copy()

    Xs, ys = zip(*_calc())

    return np.concatenate(Xs), np.concatenate(ys)


def bbox_iou(bbox1, bbox2):
    # min_row, min_col, max_row, max_col

    bbox1 = shapely.geometry.box(*bbox1)
    bbox2 = shapely.geometry.box(*bbox2)

    return bbox1.intersection(bbox2).area / bbox1.union(bbox2).area


def match_bbox_iou(regions_true, regions_pred):
    """Return best-match IoUs for each true region."""

    if not regions_pred:
        return [0 for r_true in regions_true]

    return [
        max(bbox_iou(r_true.bbox, r_pred.bbox) for r_pred in regions_pred)
        for r_true in regions_true
    ]


def evaluate_labels(labels_true, labels_pred, pos_label=1):
    mask = labels_true > 0

    y_true = labels_true[mask]
    y_pred = labels_pred[mask]

    (precision, recall, f_score, _,) = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_label
    )

    fg_true = labels_true == pos_label
    fg_pred = labels_pred == pos_label

    jaccard_score = sklearn.metrics.jaccard_score(
        fg_true.reshape(-1), fg_pred.reshape(-1)
    )

    # Calculate connected components
    # FIXME: Move this step out of the evaluation and use annotated gt objects instead of components derived from binary mask
    segments_true = skimage.measure.label(fg_true)
    segments_pred = skimage.measure.label(fg_pred)

    regions_true = skimage.measure.regionprops(segments_true)
    regions_pred = skimage.measure.regionprops(segments_pred)

    bbox_ious = match_bbox_iou(regions_true, regions_pred)

    # Homogeneity/Completeness: Make sure that the assignment of pixels to individual segments matches
    common_mask = fg_true & fg_pred
    (
        homogeneity,
        completeness,
        v_score,
    ) = sklearn.metrics.homogeneity_completeness_v_measure(
        segments_true[common_mask], segments_pred[common_mask]
    )

    return {
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "jaccard_score": jaccard_score,
        "bbox_ious": bbox_ious,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_score": v_score,
    }


def evaluate_classifier(clf, dataset, cache, *, prefix=None, post_processor=None):
    def _calc():
        for row in tqdm.tqdm(dataset.itertuples(), desc="Evaluate", total=len(dataset)):
            key = row.Index + "/features"
            features = cache[key]

            labels_true = np.array(PIL.Image.open(row.labels_fn).convert("L"))
            labels_pred = predict_segmenter(features, clf)

            if post_processor is not None:
                # FIXME: Supply image
                labels_pred = post_processor(None, labels_pred)

            metrics = evaluate_labels(labels_true, labels_pred)

            if prefix is not None:
                metrics = {prefix + k: v for k, v in metrics.items()}

            yield {"id": row.Index, **metrics}

    return pd.DataFrame(_calc()).set_index("id", drop=True)  # type: ignore


def compute_cache_location(*trials):
    data = {}
    for t in trials:
        data.update(t.todict(True))

    encoded = json.dumps(data, sort_keys=True, ensure_ascii=True).encode()
    hash = hashlib.md5(encoded)

    return hash.hexdigest()


def label2bin(labels):
    return labels == 1


def bin2label(mask):
    out = np.full(mask.shape, 2, dtype=int)
    out[mask] = 1
    return out


class ErodeDilatePostProcessor:
    def __init__(self, erode_1=0, dilate=0, erode_2=0) -> None:
        self.erode_1 = erode_1
        self.dilate = dilate
        self.erode_2 = erode_2

    def __call__(self, image, labels):
        mask = label2bin(labels)

        if self.erode_1:
            mask = skimage.morphology.binary_erosion(
                mask, skimage.morphology.disk(self.erode_1)
            )
        if self.dilate:
            mask = skimage.morphology.binary_dilation(
                mask, skimage.morphology.disk(self.dilate)
            )
        if self.erode_2:
            mask = skimage.morphology.binary_erosion(
                mask, skimage.morphology.disk(self.erode_2)
            )

        return bin2label(mask)


class WatershedPostProcessor:
    """
    Align the borders of the predicted mask to the gradient image.

    - Object seed: Eroded foreground
    - Background seed: Eroded background
    """


class ClosingPostProcessor:
    def __init__(self, radius=3) -> None:
        self.radius = radius

    def __call__(self, image, labels):
        del image

        mask = label2bin(labels)

        mask = skimage.morphology.binary_closing(
            mask, skimage.morphology.disk(self.radius)
        )

        return bin2label(mask)


@overload
def _trainval_single(
    trial: Trial, cache, fold, dataset_train, post_processor
) -> Tuple[Any, dict]:
    ...


@overload
def _trainval_single(
    trial: Trial, cache, fold, dataset_train, post_processor, dataset_val
) -> Tuple[Any, dict, dict]:
    ...


def _trainval_single(
    trial: Trial, cache, fold, dataset_train, post_processor, dataset_val=None
):
    print("Preparing training set...")
    X_train, y_train = trial.prefixed("prepare_train_").call(
        partial(prepare_training_set, dataset_train, cache)
    )
    print(f"Training set shape: {X_train.shape}")

    clf_cls = trial.choice(
        "classifier", [RandomForestClassifier, LinearSVC, DummyClassifier]
    )
    clf = trial.prefixed("classifier_").call(clf_cls)

    trial.save()

    print("Fitting classifier...")
    with Timer("train") as t_fit:
        clf.fit(X_train, y_train)

    print("Evaluating classifier (train)...")
    with Timer("eval_train") as t:
        metrics_train = evaluate_classifier(
            clf, dataset_train, cache, prefix="train_", post_processor=post_processor
        )
    time_eval_train = float(t.elapsed) / len(dataset_train)

    metrics_train.to_csv(os.path.join(trial.wdir, f"{fold}_metrics_train.csv"))

    results_train = {
        "fold": fold,
        "t_fit": float(t_fit.elapsed),
        "time_eval_train": time_eval_train,
        **metrics_train.mean(axis=0, numeric_only=True).to_dict(),
    }

    if dataset_val is None:
        return clf, results_train

    print("Evaluating classifier (val)...")
    with Timer("eval_val") as t:
        metrics_val = evaluate_classifier(
            clf, dataset_val, cache, prefix="val_", post_processor=post_processor
        )
    time_eval_val = float(t.elapsed) / len(dataset_val)

    metrics_val.to_csv(os.path.join(trial.wdir, f"{fold}_metrics_val.csv"))

    results_val = {
        "fold": fold,
        "time_eval_val": time_eval_val,
        **metrics_val.mean(axis=0, numeric_only=True).to_dict(),
    }

    return clf, results_train, results_val


@Experiment(
    active=False,
)
def trainval(trial: Trial):
    # Find dataset files
    dataset = trial.prefixed("dataset_").call(load_dataset)

    feature_extractor_cls = trial.choice(
        "feature_extractor", [MultiscaleBasicFeatures, NullFeatures]
    )

    # Extract features
    feature_extractor = trial.prefixed("feature_extractor_").call(feature_extractor_cls)

    post_processor_cls = trial.choice("post_processor", [None, ClosingPostProcessor])
    post_processor = (
        trial.prefixed("post_processor_").call(post_processor_cls)
        if post_processor_cls is not None
        else None
    )

    with H5Cache(os.path.join(trial.wdir, "features.h5"), cleanup=True) as cache:
        print("Calculating features...")

        with Timer("calculate_features") as t:
            calculate_features(dataset, cache, feature_extractor)
        time_calculate_features = float(t.elapsed) / len(dataset)

        print("Running cross-validation...")

        results_train = []
        results_val = []

        # Split (stratified by cluster)
        k = trial.get("k", 5)
        for fold, (train_index, val_index) in enumerate(
            tqdm.tqdm(
                StratifiedKFold(k).split(dataset.index, dataset["node_id"]),
                desc=f"{k}-fold cross-validation",
            )
        ):
            print(f"Validation split {fold}")
            trial.status = f"Trainval {fold}"

            dataset_train = dataset.iloc[train_index]
            dataset_val = dataset.iloc[val_index]

            _, metrics_train, metrics_val = _trainval_single(
                trial, cache, fold, dataset_train, post_processor, dataset_val
            )

            results_train.append(metrics_train)
            results_val.append(metrics_val)

    results_train = pd.DataFrame(results_train)
    results_val = pd.DataFrame(results_val)

    results_train.to_csv(os.path.join(trial.wdir, "results_train.csv"))
    results_val.to_csv(os.path.join(trial.wdir, "results_val.csv"))

    trial.status = "Finished"

    return {
        "time_calculate_features": time_calculate_features,
        **results_train.mean(axis=0, numeric_only=True).to_dict(),
        **results_val.mean(axis=0, numeric_only=True).to_dict(),
    }


trainval_base = Experiment(
    configurator=[
        Const(dataset_path="data/22-06-02-LOKI"),
    ],
    parent=trainval,
)

trainval_features = Experiment(
    configurator=[
        Grid({"feature_extractor": ["MultiscaleBasicFeatures", "NullFeatures"]}),
    ],
    parent=trainval_base,
    active=False,
)

trainval_features_rf = Experiment(
    "trainval_rf",
    parameters=[
        Const(
            classifier="RandomForestClassifier",
            classifier_n_estimators=10,
        )
    ],
    parent=trainval_features,
    defaults={"classifier_verbose": 1},
)

trainval_features_lsvc = Experiment(
    "trainval_lsvc",
    parameters=[
        Const(
            classifier="LinearSVC",
        )
    ],
    parent=trainval_features,
)

trainval_features_dummy = Experiment(
    "trainval_dummy",
    parameters=[
        Const(
            classifier="DummyClassifier",
        )
    ],
    parent=trainval_features,
)

trainval_multiscale_basic_features = Experiment(
    "trainval_multiscale_basic_features",
    configurator=[
        Const(
            classifier="RandomForestClassifier",
            classifier_n_estimators=10,
            feature_extractor="MultiscaleBasicFeatures",
        ),
        AdditiveConfiguratorChain(
            # sigma_max
            Grid({"feature_extractor_sigma_max": [4, 8, 16, 32, 64]}),
            # sigma_min
            Grid({"feature_extractor_sigma_min": [0.5, 1, 2]}),
            # features
            Grid(
                {
                    "feature_extractor_edges": [True, False],
                    "feature_extractor_intensity": [True, False],
                    "feature_extractor_texture": [True, False],
                }
            )
            * FilterConfig(
                lambda p:
                # At least one feature must be true
                any(
                    p[k]
                    for k in [
                        "feature_extractor_edges",
                        "feature_extractor_intensity",
                        "feature_extractor_texture",
                    ]
                )
            ),
            shuffle=True,
        ),
    ],
    parent=trainval_base,
)

trainval_label_augmentation = Experiment(
    "trainval_label_augmentation",
    configurator=[
        Const(
            classifier="RandomForestClassifier",
            classifier_n_estimators=10,
            feature_extractor="MultiscaleBasicFeatures",
        ),
        # Initial config + current optimum
        Const(prepare_train_bg_margin=0, prepare_train_bg_width=0)
        + Const(prepare_train_bg_margin=7, prepare_train_bg_width=7),
        AdditiveConfiguratorChain(
            Const(),
            # RandomGrid(
            #     {"classifier_class_weight": [None, "balanced", "balanced_subsample"]}
            # ),
            RandomGrid(
                {
                    "prepare_train_bg_margin": [0, 1, 3, 7, 15],
                    "prepare_train_bg_width": [1, 3, 7, 15, 31, 63],
                }
            ),
            shuffle=True,
        ),
    ],
    parent=trainval_base,
)

trainval_post_processing = Experiment(
    "trainval_post_processing",
    configurator=[
        Const(dataset_path="data/22-06-02-LOKI"),
        # Const(dataset_path="data/22-05-25-LOKI"),
        Const(
            classifier="RandomForestClassifier",
            classifier_n_estimators=10,
            feature_extractor="MultiscaleBasicFeatures",
            feature_extractor_sigma_max=32,
            feature_extractor_edges=True,
            feature_extractor_intensity=True,
            feature_extractor_texture=True,
            # Label augmentation for close fit to the true annotation (jaccard score)
            prepare_train_bg_margin=3,
            prepare_train_bg_width=15,
        ),
        Const()
        + (
            Const(post_processor="ClosingPostProcessor")
            * RandomGrid({"post_processor_radius": [1, 3, 5, 9, 15, 31]})
        ),
    ],
    parent=trainval,
    defaults={"n_jobs": 4},
)

trainval_n_estimators = Experiment(
    "trainval_n_estimators",
    configurator=[
        Const(dataset_path="data/22-06-02-LOKI"),
        Const(
            classifier="RandomForestClassifier",
            classifier_n_estimators=10,
            feature_extractor="MultiscaleBasicFeatures",
            feature_extractor_sigma_max=32,
            feature_extractor_edges=True,
            feature_extractor_intensity=True,
            feature_extractor_texture=True,
            # Label augmentation for close fit to the true annotation (jaccard score)
            prepare_train_bg_margin=3,
            prepare_train_bg_width=15,
            post_processor="ClosingPostProcessor",
            post_processor_radius=1,
        ),
        RandomGrid({"classifier_n_estimators": [10, 25, 50, 100]}),
    ],
    parent=trainval,
    defaults={"n_jobs": 4},
)

trainval_min_intensity = Experiment(
    "trainval_min_intensity",
    configurator=[
        Const(dataset_path="data/22-06-02-LOKI"),
        Const(
            classifier="RandomForestClassifier",
            classifier_n_estimators=10,
            feature_extractor="MultiscaleBasicFeatures",
            feature_extractor_sigma_max=32,
            feature_extractor_edges=True,
            feature_extractor_intensity=True,
            feature_extractor_texture=True,
            # Label augmentation for close fit to the true annotation (jaccard score)
            prepare_train_bg_margin=3,
            prepare_train_bg_width=15,
            post_processor="ClosingPostProcessor",
            post_processor_radius=1,
        ),
        RandomGrid({"prepare_train_min_intensity": [0, 16, 24, 32, 64]}),
    ],
    parent=trainval,
    defaults={"n_jobs": 4},
)

###############################################################################


@Experiment(
    configurator=[
        Const(dataset_path="data/22-06-02-LOKI"),
        Const(
            classifier="RandomForestClassifier",
            classifier_n_estimators=10,
            feature_extractor="MultiscaleBasicFeatures",
            feature_extractor_sigma_max=32,
            feature_extractor_edges=True,
            feature_extractor_intensity=True,
            feature_extractor_texture=True,
            # Label adaptation for close fit to the true annotation (jaccard score)
            prepare_train_bg_margin=3,
            prepare_train_bg_width=15,
            prepare_train_min_intensity=24,
        ),
    ],
    defaults={
        "classifier_verbose": 2,
        "n_jobs": -1,
        ###
        ## Influences only the evaluation
        # Closing with r=1 to increase recall (and completeness) - without harming the others too much
        "post_processor": "ClosingPostProcessor",
        "post_processor_radius": 1,
    },
)
def train(trial: Trial):
    # Find dataset files
    dataset = trial.prefixed("dataset_").call(load_dataset)

    feature_extractor_cls = trial.choice(
        "feature_extractor", [MultiscaleBasicFeatures, NullFeatures]
    )

    # Extract features
    feature_extractor = trial.prefixed("feature_extractor_").call(feature_extractor_cls)

    post_processor_cls = trial.choice("post_processor", [None, ClosingPostProcessor])
    post_processor = (
        trial.prefixed("post_processor_").call(post_processor_cls)
        if post_processor_cls is not None
        else None
    )

    cache_fn = os.path.join(trial.wdir, "features.h5")

    with H5Cache(cache_fn, cleanup=True) as cache:
        print("Calculating features...")

        with Timer("calculate_features") as t:
            calculate_features(dataset, cache, feature_extractor)
        time_calculate_features = float(t.elapsed) / len(dataset)

        clf, metrics_train = _trainval_single(trial, cache, 0, dataset, post_processor)

    classifier_fn = os.path.join(trial.wdir, "classifier.pkl")
    with open(classifier_fn, "wb") as f:
        pickle.dump(clf, f)

    print(classifier_fn)

    return {
        "time_calculate_features": time_calculate_features,
        **metrics_train,
    }

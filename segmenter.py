import inspect
from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import scipy.ndimage as ndi
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation

import isotropic


class DefaultReprMixin:
    def __repr__(self) -> str:
        params = [
            f"{p}={getattr(self, p)!r}"
            for p in inspect.signature(type(self)).parameters.keys()
            if hasattr(self, p)
        ]
        return self.__class__.__name__ + "(" + (", ".join(params)) + ")"


class FeatureExtractor(DefaultReprMixin, ABC):
    @abstractmethod
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

        self.n_jobs = 1

    def __call__(self, image):
        return skimage.feature.multiscale_basic_features(
            image,
            intensity=self.intensity,
            edges=self.edges,
            texture=self.texture,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            num_sigma=self.num_sigma,
            num_workers=self.n_jobs,
        )


class NullFeatures(FeatureExtractor):
    def __call__(self, image):
        # Ensure HxWxC
        return image.reshape(image.shape[:2] + (-1,))


class PreSelector(DefaultReprMixin, ABC):
    """
    Generate a mask from an image.

    The simplest implementation would be thresholding.
    """

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class MinIntensityPreSelector(PreSelector):
    def __init__(self, min_intensity, dilate=0) -> None:
        super().__init__()
        self.min_intensity = min_intensity
        self.dilate = dilate

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 2

        mask = image > self.min_intensity

        if self.dilate:
            mask = isotropic.isotropic_dilation(mask, self.dilate)

        return mask


class PostProcessor(DefaultReprMixin, ABC):
    """
    Post-process the classifier output to obtain a labeled image.

    This can include thresholding, morphological operations and labeling.
    """

    @abstractmethod
    def __call__(self, scores: np.ndarray, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


def _label_ex(
    mask_pred,
    image,
    *,
    min_size=0,
    closing=0,
    relative_closing=0,
    min_intensity=0,
    clear_background: Optional[np.ndarray] = None,
):
    labels_pred = skimage.measure.label(mask_pred)

    if min_size or min_intensity:
        # Remove elements with maximum intensity smaller than min_intensity or area smaller than min_size
        for r in skimage.measure.regionprops(labels_pred, image):
            if (r.intensity_max < min_intensity) or (r.area < min_size):
                labels_pred[labels_pred == r.label] = 0

    mask_pred = labels_pred > 0

    if relative_closing:
        # Close relative to maximum inner object diameter and relabel
        dist = ndi.distance_transform_edt(mask_pred)
        closing += dist.max() * relative_closing

    if closing:
        # Close and relabel
        mask_pred = isotropic.isotropic_closing(mask_pred, closing)

        if clear_background is not None:
            mask_pred &= ~clear_background

        labels_pred = skimage.measure.label(mask_pred)

    return labels_pred


class DefaultPostProcessor(PostProcessor):
    def __init__(
        self,
        threshold=0.5,
        smoothing=0,
        min_size=0,
        closing=0,
        min_intensity=0,
        relative_closing=0,
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.smoothing = smoothing
        self.min_size = min_size
        self.closing = closing
        self.min_intensity = min_intensity
        self.relative_closing = relative_closing

    def __call__(self, scores: np.ndarray, image: np.ndarray) -> np.ndarray:
        if self.smoothing:
            skimage.filters.gaussian(scores, self.smoothing)

        mask_pred = scores > self.threshold

        return _label_ex(
            mask_pred,
            image,
            min_size=self.min_size,
            closing=self.closing,
            min_intensity=self.min_intensity,
            relative_closing=self.relative_closing,
        )


class WatershedPostProcessor(PostProcessor):
    """
    Post-Processing of predicted scores based on Watershed.

    Args:
        thr_low: Scores below this value are background.
        q_high: Scores above this quantile are foreground.
        min_intensity: Only retain a segment if any part is > min_intensity
        open_background: Open background so that small background areas are deleted.
    """

    def __init__(
        self,
        thr_low=0.5,
        q_low=0.5,
        thr_high=0.9,
        q_high=0.99,
        edges: Literal["image", "scores"] = "image",
        dilate_edges=5,
        min_size=64,
        closing=10,
        relative_closing=0,
        min_intensity=64,
        clear_background=False,
        open_background=5,
        score_sigma=5,
    ) -> None:
        super().__init__()

        self.thr_low = thr_low
        self.q_low = q_low
        self.thr_high = thr_high
        self.q_high = q_high
        self.edges = edges
        self.dilate_edges = dilate_edges
        self.min_size = min_size
        self.closing = closing
        self.relative_closing = relative_closing
        self.min_intensity = min_intensity
        self.clear_background = clear_background
        self.open_background = open_background
        self.score_sigma = score_sigma

    def _edges(self, image: np.ndarray):
        if self.dilate_edges:
            image = skimage.morphology.dilation(
                image, skimage.morphology.disk(self.dilate_edges)
            )
        return skimage.filters.sobel(image)

    def __call__(self, scores: np.ndarray, image: np.ndarray) -> np.ndarray:
        if self.score_sigma:
            # Smooth scores smooth the quantile calculation
            scores = skimage.filters.gaussian(scores, self.score_sigma)

        thr_low_, thr_high_ = np.quantile(scores, [self.q_low, self.q_high])

        # Take maximum of thr_low and q_low to ensure that at least a bit background is visible
        thr_low = max(self.thr_low, thr_low_)
        # Take minimum of thr_high and q_high quantile to ensure that at least some object is visible
        thr_high = min(self.thr_high, thr_high_)

        markers = np.zeros(scores.shape, dtype="uint8")
        FOREGROUND, BACKGROUND = 1, 2

        mask_foreground = scores > thr_high
        markers[mask_foreground] = FOREGROUND

        mask_background = scores < thr_low
        if self.open_background:
            mask_background = isotropic.isotropic_opening(
                mask_background, self.open_background
            )
        markers[mask_background] = BACKGROUND

        if self.edges == "image":
            edges = self._edges(image)
        elif self.edges == "scores":
            edges = self._edges(scores)
        else:
            raise ValueError(f"Unknown edges: {self.edges!r}")

        ws = skimage.segmentation.watershed(edges, markers)
        mask_pred = ws == FOREGROUND

        clear_background = ws == BACKGROUND if self.clear_background else None

        return _label_ex(
            mask_pred,
            image,
            min_size=self.min_size,
            closing=self.closing,
            relative_closing=self.relative_closing,
            min_intensity=self.min_intensity,
            clear_background=clear_background,
        )


class Segmenter(DefaultReprMixin):
    """
    Segmenter for images.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        classifier,
        postprocessor: PostProcessor,
        preselector: Optional[PreSelector] = None,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.postprocessor = postprocessor
        self.preselector = preselector

    def __call__(self, image: np.ndarray):
        """Apply the full segmentation to the image and return a label image."""
        features = self.extract_features(image)
        mask = self.preselect(image)
        mask = self.predict_pixels(features, mask)
        return self.postprocess(mask, image)

    def extract_features(self, image: np.ndarray):
        return self.feature_extractor(image)

    def preselect(self, image: np.ndarray):
        if self.preselector is None:
            return None
        return self.preselector(image)

    def predict_pixels(
        self, features: np.ndarray, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        # features is [h,w,c]
        h, w, c = features.shape

        if mask is not None:
            # Predict only masked locations and assemble result
            prob = self.classifier.predict_proba(features[mask])[:, 1]
            result = np.zeros((h, w), dtype=prob.dtype)
            result[mask] = prob
            return result

        # Return probability of foreground in the same shape as the input
        return self.classifier.predict_proba(features)[:, 1].reshape((h, w))

    def postprocess(self, mask: np.ndarray, image: np.ndarray):
        return self.postprocessor(mask, image)

    def configure(self, **kwargs):
        """
        Configure the underlying objects.

        Arguments include n_jobs, verbose, ...
        """
        for name in ("feature_extractor", "classifier", "postprocessor", "preselector"):
            obj = getattr(self, name)
            for k, v in kwargs.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)

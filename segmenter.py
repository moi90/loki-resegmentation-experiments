from abc import ABC, abstractmethod
from optparse import Option
from typing import Optional
import numpy as np
import skimage.feature
import skimage.measure
import skimage.morphology
import skimage.filters
import skimage.segmentation
import inspect
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


def _label_ex(mask_pred, image, min_size=0, closing=0, min_intensity=0):
    labels_pred = skimage.measure.label(mask_pred)

    if min_size or min_intensity:
        # Remove elements with maximum intensity smaller than min_intensity or area smaller than min_size
        for r in skimage.measure.regionprops(labels_pred, image):
            if (r.intensity_max < min_intensity) or (r.area < min_size):
                labels_pred[labels_pred == r.label] = 0

    if closing:
        # Close and relabel
        mask_pred = isotropic.isotropic_closing(labels_pred > 0, closing)
        labels_pred = skimage.measure.label(mask_pred)

    return labels_pred


class DefaultPostProcessor(PostProcessor):
    def __init__(
        self, threshold=0.5, smoothing=0, min_size=0, closing=0, min_intensity=0
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.smoothing = smoothing
        self.min_size = min_size
        self.closing = closing
        self.min_intensity = min_intensity

    def __call__(self, scores: np.ndarray, image: np.ndarray) -> np.ndarray:
        if self.smoothing:
            skimage.filters.gaussian(scores, self.smoothing)

        mask_pred = scores > self.threshold

        return _label_ex(
            mask_pred, image, self.min_size, self.closing, self.min_intensity
        )


class WatershedPostProcessor(PostProcessor):
    """
    Post-Processing of predicted scores based on Watershed.

    Args:
        thr_low: Scores below this value are background.
        q_high: Scores above this quantile are foreground.
        min_intensity: Only retain a segment if any part is > min_intensity
    """

    def __init__(
        self,
        thr_low=0.5,
        q_high=0.99,
        dilate_edges=3,
        min_size=64,
        closing=10,
        min_intensity=64,
    ) -> None:
        super().__init__()

        self.thr_low = thr_low
        self.q_high = q_high
        self.dilate_edges = dilate_edges
        self.min_size = min_size
        self.closing = closing
        self.min_intensity = min_intensity

    def _edges(self, image: np.ndarray):
        if self.dilate_edges:
            image = skimage.morphology.dilation(
                image, skimage.morphology.disk(self.dilate_edges)
            )
        return skimage.filters.sobel(image)

    def __call__(self, scores: np.ndarray, image: np.ndarray) -> np.ndarray:
        thr_high = np.quantile(scores, self.q_high)
        markers = np.zeros(scores.shape, dtype="uint8")
        FOREGROUND, BACKGROUND = 1, 2

        markers[scores > thr_high] = FOREGROUND
        markers[scores < self.thr_low] = BACKGROUND

        edges = self._edges(image)

        mask_pred = skimage.segmentation.watershed(edges, markers) == FOREGROUND

        return _label_ex(
            mask_pred, image, self.min_size, self.closing, self.min_intensity
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

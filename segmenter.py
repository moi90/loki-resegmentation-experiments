from abc import ABC, abstractmethod
from optparse import Option
from typing import Optional
import numpy as np
import skimage.feature
import skimage.measure


class FeatureExtractor(ABC):
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


class PreSelector(ABC):
    """
    Generate a mask from an image.

    The simplest implementation would be thresholding.
    """

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class MinIntensityPreSelector(PreSelector):
    def __init__(self, min_intensity) -> None:
        super().__init__()
        self.min_intensity = min_intensity

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 2
        return image > self.min_intensity


class PostProcessor(ABC):
    """
    Post-process the classifier output to obtain a labeled image.

    This can include thresholding, morphological operations and labeling.
    """

    @abstractmethod
    def __call__(self, scores: np.ndarray, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class DefaultPostProcessor(PostProcessor):
    def __call__(self, scores: np.ndarray, image: np.ndarray) -> np.ndarray:
        del image
        mask = scores > 0.5
        return skimage.measure.label(mask)


class Segmenter:
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

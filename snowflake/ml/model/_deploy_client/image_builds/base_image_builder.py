from abc import ABC, abstractmethod


class ImageBuilder(ABC):
    """
    Abstract class encapsulating image building and upload to model registry.
    """

    @abstractmethod
    def build_and_upload_image(self) -> str:
        """Builds and uploads an image to the model registry.

        Returns: Full image path.
        """
        pass

from snowflake.ml.model._deploy_client.docker import base_image_builder


class ClientImageBuilder(base_image_builder.ImageBuilder):
    """
    Class for client-side image building and upload to model registry.
    """

    def build_and_upload_image(self) -> None:
        """
        Builds and uploads an image to the model registry.
        TODO: Actual implementation coming.
        """
        self._build()
        self._upload()

    def _build(self) -> None:
        """
        Builds image in client side.
        TODO: Actual implementation coming.
        """
        pass

    def _upload(self) -> None:
        """
        Uploads image to image registry.
        TODO: Actual implementation coming.
        """
        pass

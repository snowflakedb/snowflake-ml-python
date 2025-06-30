from typing import Any, Iterator, Optional, Union

import numpy as np
import numpy.typing as npt
import torch.utils.data

from snowflake.ml.data import data_ingestor


class TorchDatasetWrapper(torch.utils.data.IterableDataset[dict[str, Any]]):
    """Wrap a DataIngestor into a PyTorch IterableDataset"""

    def __init__(
        self,
        ingestor: data_ingestor.DataIngestor,
        *,
        batch_size: Optional[int],
        shuffle: bool = False,
        drop_last: bool = False,
        expand_dims: bool = True,
    ) -> None:
        """Not intended for direct usage. Use DataConnector.to_torch_dataset() instead"""
        squeeze = False
        if batch_size is None:
            batch_size = 1
            squeeze = True

        self._ingestor = ingestor
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._squeeze_outputs = squeeze
        self._expand_dims = expand_dims

    def __iter__(self) -> Iterator[dict[str, Union[npt.NDArray[Any], list[Any]]]]:
        max_idx = 0
        filter_idx = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            max_idx = worker_info.num_workers - 1
            filter_idx = worker_info.id

        if self._shuffle and worker_info is not None:
            raise RuntimeError("Dataset shuffling not currently supported with multithreading")

        counter = 0
        for batch in self._ingestor.to_batches(
            batch_size=self._batch_size, shuffle=self._shuffle, drop_last_batch=self._drop_last
        ):
            # Skip indices during multi-process data loading to prevent data duplication
            if counter == filter_idx:
                yield {
                    k: _preprocess_array(v, squeeze=self._squeeze_outputs, expand_dims=self._expand_dims)
                    for k, v in batch.items()
                }
            if counter < max_idx:
                counter += 1
            else:
                counter = 0


class TorchDataPipeWrapper(TorchDatasetWrapper, torch.utils.data.IterDataPipe[dict[str, Any]]):
    """Wrap a DataIngestor into a PyTorch IterDataPipe"""

    def __init__(
        self,
        ingestor: data_ingestor.DataIngestor,
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        expand_dims: bool = True,
    ) -> None:
        """Not intended for direct usage. Use DataConnector.to_torch_datapipe() instead"""
        super().__init__(ingestor, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, expand_dims=expand_dims)


def _preprocess_array(
    arr: npt.NDArray[Any], squeeze: bool = False, expand_dims: bool = True
) -> Union[npt.NDArray[Any], list[np.object_]]:
    """Preprocesses batch column values."""
    single_dimensional = arr.ndim < 2 and not arr.dtype == np.object_

    # Squeeze away all extra dimensions. This is only used when batch_size = None.
    if squeeze:
        arr = arr.squeeze(axis=0)

    # For single dimensional data,
    if single_dimensional and expand_dims:
        axis = 0 if arr.ndim == 0 else 1
        arr = np.expand_dims(arr, axis=axis)

    # Handle object arrays.
    if arr.dtype == np.object_:
        array_list = arr.tolist()
        # If this is an array of arrays, convert the dtype to match the underlying array.
        # Otherwise, if this is a numpy array of strings, convert the array to a list.
        arr = np.array(array_list, dtype=arr.item(0).dtype) if isinstance(arr.item(0), np.ndarray) else array_list

    return arr

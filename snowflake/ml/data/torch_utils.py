from typing import Any, Dict, Iterator, List, Union

import numpy as np
import numpy.typing as npt
import torch.utils.data

from snowflake.ml.data import data_ingestor


class TorchDatasetWrapper(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """Wrap a DataIngestor into a PyTorch IterableDataset"""

    def __init__(
        self,
        ingestor: data_ingestor.DataIngestor,
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        squeeze_outputs: bool = True
    ) -> None:
        """Not intended for direct usage. Use DataConnector.to_torch_dataset() instead"""
        self._ingestor = ingestor
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._squeeze_outputs = squeeze_outputs

    def __iter__(self) -> Iterator[Dict[str, Union[npt.NDArray[Any], List[Any]]]]:
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
                # Basic preprocessing on batch values: squeeze away extra dimensions
                # and convert object arrays (e.g. strings) to lists
                if self._squeeze_outputs:
                    yield {
                        k: (v.squeeze().tolist() if v.dtype == np.object_ else v.squeeze()) for k, v in batch.items()
                    }
                else:
                    yield batch  # type: ignore[misc]

            if counter < max_idx:
                counter += 1
            else:
                counter = 0


class TorchDataPipeWrapper(TorchDatasetWrapper, torch.utils.data.IterDataPipe[Dict[str, Any]]):
    """Wrap a DataIngestor into a PyTorch IterDataPipe"""

    def __init__(
        self, ingestor: data_ingestor.DataIngestor, *, batch_size: int, shuffle: bool = False, drop_last: bool = False
    ) -> None:
        """Not intended for direct usage. Use DataConnector.to_torch_datapipe() instead"""
        super().__init__(ingestor, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, squeeze_outputs=False)

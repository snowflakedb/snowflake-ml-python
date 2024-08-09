from typing import Any, Dict, Iterator

import torch.utils.data

from snowflake.ml.data import data_ingestor


class TorchDataset(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """Implementation of PyTorch IterableDataset"""

    def __init__(self, ingestor: data_ingestor.DataIngestor, shuffle: bool = False) -> None:
        """Not intended for direct usage. Use DataConnector.to_torch_dataset() instead"""
        self._ingestor = ingestor
        self._shuffle = shuffle

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        max_idx = 0
        filter_idx = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            max_idx = worker_info.num_workers - 1
            filter_idx = worker_info.id

        counter = 0
        for batch in self._ingestor.to_batches(batch_size=1, shuffle=self._shuffle, drop_last_batch=False):
            # Skip indices during multi-process data loading to prevent data duplication
            if counter == filter_idx:
                yield {k: v.item() for k, v in batch.items()}

            if counter < max_idx:
                counter += 1
            else:
                counter = 0

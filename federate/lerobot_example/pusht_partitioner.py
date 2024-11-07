ds.filter(lambda x: x["label"] == 1)

# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
# Copyright zk0 DBA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Partitioner class that works with the Hugging Face lerobot/pusht Dataset."""


import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class PushtPartitioner(Partitioner):
    """Partitioner creates each partition with even number of task samples using episode_index % num_partitions = parition_id.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import IidPartitioner
    >>>
    >>> partitioner = PushtPartitioner(num_partitions=10)
    >>> fds = FederatedDataset(dataset="lerobot/pusht", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    """

    def __init__(self, num_partitions: int) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        return self.dataset.filter((lambda x: x["episode_index"] % self._num_partitions == partition_id))

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        return self._num_partitions

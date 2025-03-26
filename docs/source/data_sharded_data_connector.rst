:orphan:

.. #

    This file is temporary until the snowflake.ml.modeling.distributors subpackage makes it into snowflake-ml-python

****************************
Sharded Data Connector
****************************

When using Container Runtime for ML in a Snowflake Notebook, a sharded data connector is available to allow each process in distributed training to read a subset of the data.

.. _label-sharded-data:

:code:`snowflake.ml.data_sharded_data_connector.ShardedDataConnector`
=====================================================================

A data connector subclass that is used to shard data for distributed training.

Data is sharded automatically into the number of partitions that matches the ``world_size`` of the distributed trainer. Call
``get_shard`` within a Snowflake training context to retrieve the shard associated with that worker process.

Example usage:

.. code-block:: python

    # Load from Snowpark Dataframe
    df = session.table("TRAIN_DATA_TABLE")
    train_data = ShardedDataConnector.from_dataframe(df)

    # Pass to pytorch trainer to retrieve shard in training function.
    def train_func():
        dataset_map = context.get_dataset_map()
        training_data = dataset_map["train"].get_shard().to_torch_dataset()

    pytroch_trainer = PyTorchTrainer(
        train_func=train_func,
    )

    pytroch_trainer.run(
        dataset_map=dict(
            train=train_data
        )
    )

Methods
    ``classmethod from_dataframe``
        Creates a sharded data connector from a Snowpark DataFrame.

        Args
            ``df -> Snowpark DataFrame``
                The Snowpark ``DataFrame`` containing the data to shard.

            ``ingestor_class: -> DataIngestor``
                ``DataIngestor`` class to use for reading the dataset.

            ``equal -> bool``
                If True, each shard has the same number of rows. Some rows may be dropped. If False, each shard has a roughly equal number of rows, but some shards may have more rows than others.

    ``classmethod from_dataset``
        Creates a sharded data connector from a Snowflake Dataset.

        Args
            ``ds -> dataset_dataset``
                Dataset to be read and sharded.

            ``ingestor_class --> DataIngestor``
                ``DataIngestor`` class to use for reading the dataset.

            ``equal -> bool``
                If True, each shard has the same number of rows. Some rows may be dropped. If False, each shard has a roughly equal number of rows, but some shards may have more rows than others.

    ``classmethod from_sources``
        Creates a sharded data connector from a list of Snowflake DataSources. A DataSource may be either a Snowpark DataFrame or a Dataset.

        Args
            ``sources -> List[DataSource]``
                List of ``DataSource``s to be read and sharded.

            ``ingestor_class --> DataIngestor``
                ``DataIngestor`` class to use for reading the dataset.

            ``equal -> bool``
                If ``True``, each shard has the same number of rows. Some rows may be dropped. If False, each shard has a roughly equal number of rows, but some shards may have more rows than others.

    ``get_shard -> DataConnector``
        Retrieves the shard of data associated with the rank of the calling process, allowing each process to retrieve its specific shard.

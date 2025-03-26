:orphan:

.. #

    This file is temporary until the snowflake.ml.modeling.distributors subpackage makes it into snowflake-ml-python

****************************
Distributed Modeling Classes
****************************

When using `Container Runtime for ML <https://docs.snowflake.com/user-guide/ui-snowsight/notebooks-on-spcs>`_
in a `Snowflake Notebook <https://docs.snowflake.com/user-guide/ui-snowsight/notebooks>`_, a set of distributed
modeling classes is available to train selected types of models on large datasets using the full resources of a
Snowpark Container Services (SPCS) `compute pool <https://docs.snowflake.com/en/developer-guide/snowflake-cli/services/manage-compute-pools>`_.

The following model types are supported:

- :ref:`XGBoost <label-distributors_xgboost>`
- :ref:`LightGBM <label-distributors_lightgbm>`
- :ref:`PyTorch <label-distributors_pytorch>`

.. _label-distributors_xgboost:

:code:`snowflake.ml.modeling.distributors.xgboost.XGBEstimator`
===============================================================

Xgboost Estimator that supports distributed training.

Arguments
    ``n_estimators (int)``
        Number of estimators. Default is 100.

    ``objective (str)``
        The objective function used for training. 'reg:squarederror'[Default] for regression,
        'binary:logistic' for binary classification, 'multi:softmax' for multi-class classification.

    ``params (Optional[dict])``
        Additional parameters for the XGBoost Estimator.

        Some key parameters are:

        - ``booster``: Specify which booster to use: ``gbtree`` (the default), ``gblinear`` or ``dart``.
        - ``max_depth``: Maximum depth of a tree. Default is 6.
        - ``max_leaves``: Maximum number of nodes to be added. Default is 0.
        - ``max_bin:`` Maximum number of bins that continuous feature values will be bucketed in. Default is 256.
        - ``eval_metric``: Evaluation metrics for validation data.

        A full list of supported parameter can be found at https://xgboost.readthedocs.io/en/stable/parameter.html
        If params dict contains keys ``n_estimators`` or ``objective``, they override the value provided
        by ``n_estimators`` and ``objective`` arguments.

    ``scaling_config (Optional[XGBScalingConfig])``
        Scaling config for XGBoost Estimator.  Defaults to None. If None, the estimator will use all available
        resources.

Related classes
---------------

``XGBScalingConfig(BaseScalingConfig)``
    Scaling config for XGBoost Estimator

    Attributes
        ``num_workers (int)``
            Number of workers to use for distributed training. Default is -1, meaning the estimator will
            use all available workers.

        ``num_cpu_per_worker (int)``
            Number of CPU cores to use per worker. Default is -1, meaning the estimator will use
            all available CPU cores.

        ``use_gpu (Optional[bool])``
            Whether to use GPU for training. If None, the estimator will choose to use GPU or not
            based on the environment.

.. _label-distributors_lightgbm:

:code:`snowflake.ml.modeling.distributors.lightgbm.LightGBMEstimator`
=====================================================================

LightGBM Estimator for distributed training and inference.

Arguments
    ``n_estimators (int, optional)``
        Number of boosting iterations. Defaults to 100.

    ``objective (str, optional)``
        The learning task and corresponding objective. Defaults to "regression".

        "regression"[Default] for regression tasks, "binary" for binary classification, "multiclass" for
        multi-class classification.

    ``params (Optional[Dict[str, Any]], optional)``
        Additional parameters for LightGBM. Defaults to None.

        Some key params are:

        - ``boosting``: The type of boosting to use. "gbdt"[Default] for Gradient Boosting Decision Tree, "dart" for
          Dropouts meet Multiple Additive Regression Trees.
        - ``num_leaves``: The maximum number of leaves in one tree. Default is 31.
        - ``max_depth``: The maximum depth of the tree. Default is -1, which means no limit.
        - ``early_stopping_rounds``: Activates early stopping. The model will train until the validation score
          stops improving. Default is 0, meaning no early stopping.

        A full list of supported parameters can be found at https://lightgbm.readthedocs.io/en/latest/Parameters.html.

    ``scaling_config (Optional[LightGBMScalingConfig], optional)``
        Configuration for scaling. Defaults to None. If None, the estimator will use all available resources.

Related classes
---------------

``LightGBMScalingConfig(BaseScalingConfig)``
    Scaling config for LightGBM Estimator.

    Attributes
        ``num_workers (int)``
            The number of worker processes to use. Default is -1, which utilizes all available resources.

        ``num_cpu_per_worker (int)``
            Number of CPUs allocated per worker. Default is -1, which means all available resources.

        ``use_gpu (Optional[bool])``
            Whether to use GPU for training. Default is None, allowing the estimator to choose
            automatically based on the environment.

.. _label-distributors_pytorch:

:code:`snowflake.ml.modeling.distributors.pytorch.PyTorchDistributor`
=====================================================================

Enables users to run distributed training with PyTorch on ContainerRuntime cluster.

PyTorchDistributor is responsible for setting up the environment, scheduling the training processes,
managing the communication between the processes, and collecting the results.

Arguments
    ``train_func (Callable)``
        A callable object that defines the training logic to be executed.

    ``scaling_config (PyTorchScalingConfig)``
        Configuration for scaling and other settings related to the training job.

Related classes
---------------

``snowflake.ml.modeling.distributors.pytorch.PyTorchScalingConfig``
    Scaling configuration for training PyTorch models.

    This class defines the scaling configuration for a PyTorch training job,
    including the number of nodes, the number of workers per node, and the
    resource requirements for each worker.

    Attributes
        ``num_nodes (int)``
            The number of nodes to use for training.

        ``num_workers_per_node (int)``
            The number of workers to use per node.

        ``resource_requirements_per_worker (WorkerResourceConfig)``
            The resource requirements for each worker, such as the number of CPUs and GPUs.

``snowflake.ml.modeling.distributors.pytorch.WorkerResourceConfig``
    Resource requirements per worker.

    This class defines the resource requirements for each worker in a distributed
    training job, specifying the number of CPU and GPU resources to allocate.

    Attributes
        ``num_cpus (int)``
            The number of CPU cores to reserve for each worker.

        ``num_gpus (int)``
            The number of GPUs to reserve for each worker. Default is 0, indicating no GPUs are reserved.

``snowflake.ml.modeling.distributors.pytorch.Context``
    Context for setting up the PyTorch distributed environment for training scripts.

    Context defines the necessary methods to manage and retrieve information
    about the distributed training environment, including worker and node ranks,
    world size, and backend configurations.

    Definitions
        Node
            A physical instance or a container.
        Worker
            A worker process in the context of distributed training.
        WorkerGroup
            The set of workers that execute the same function (e.g., trainers).
        LocalWorkerGroup
            A subset of the workers in the worker group running on the same node.
        RANK
            The rank of the worker within a worker group.
        WORLD_SIZE
            The total number of workers in a worker group.
        LOCAL_RANK
            The rank of the worker within a local worker group.
        LOCAL_WORLD_SIZE
            The size of the local worker group.
        rdzv_id
            An ID that uniquely identifies the worker group for a job. This ID is used by each node to join as
            a member of a particular worker group.
        rdzv_backend
            The backend of the rendezvous (e.g., c10d). This is typically a strongly consistent key-value store.
        rdzv_endpoint
            The rendezvous backend endpoint; usually in the form <host>:<port>.

    Methods
        ``get_world_size(self) -> int``
            Return the number of workers (or processes) participating in the job.

            For example, if training is running on 2 nodes (servers) each with 4 GPUs,
            then the world size is 8 (2 nodes * 4 GPUs per node). Usually, each GPU corresponds
            to a training process.

        ``get_rank(self) -> int``
            Return the rank of the current process across all processes.

            Rank is the unique ID given to a process to identify it uniquely across the world.
            It should be a number between 0 and world_size - 1.

            Some frameworks also call it world_rank, to distinguish it from local_rank.
            For example, if training is running on 2 nodes (servers) each with 4 GPUs,
            then the ranks will be [0, 1, 2, 3, 4, 5, 6, 7], i.e., from 0 to world_size - 1.

        ``get_local_rank(self) -> int``
            Return the local rank for the current worker.

            Local rank is a unique local ID for a worker (or process) running on the current node.

            For example, if training is running on 2 nodes (servers) each with 4 GPUs, then
            local rank for workers(or processes) running on node 0 will be [0, 1, 2, 3] and
            similarly four workers(or processes) running on node 1 will have local_rank [0, 1, 2, 3].

        ``get_local_world_size(self) -> int``
            Return the number of workers running in the current node.

            For example, if training is running on 2 nodes (servers) each with 4 GPUs,
            then local_world_size will be 4 for all processes on both nodes.

        ``get_node_rank(self)``
            Return the rank of the current node across all nodes.

            Node rank is a unique ID given to each node to identify it uniquely across all nodes
            in the world.

            For example, if training is running on 2 nodes (servers) each with 4 GPUs,
            then node ranks will be [0, 1] respectively.

        ``get_master_addr(self) -> str``
            Return IP address of the master node.

            This is typically the address of the node with node_rank 0.

        ``def get_master_port(self) -> int``
            Return port on master_addr that hosts the rendezvous server.

        ``get_default_backend(self) -> str``
            Return default backend selected by MCE.

        ``get_supported_backends(self) -> List[str]``
            Return list of supported backends by MCE.

        ``get_hyper_params(self) -> Optional[Dict[str, str]]``
            Return hyperparameter map provided to trainer.run(...) method.

        ``get_dataset_map(self) -> Optional[Dict[str, Type[DataConnector]]]``
            Return dataset map provided to trainer.run(...) method.

Related functions
-----------------

``snowflake.ml.modeling.distributors.pytorch.get_context``
    Fetches the context object that contains the worker specific runtime information.

    Returns
        ``Context``
            An instance of the Context interface that provides methods for managing the distributed training environment.

    Raises
        ``RuntimeError``
            If the PyTorch context is not available.

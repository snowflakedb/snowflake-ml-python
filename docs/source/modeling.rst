

=====================
snowflake.ml.modeling
=====================

.. automodule:: snowflake.ml.modeling
    :noindex:

snowflake.ml.modeling.calibration
---------------------------------

.. currentmodule:: snowflake.ml.modeling.calibration

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    CalibratedClassifierCV

snowflake.ml.modeling.cluster
---------------------------------

.. currentmodule:: snowflake.ml.modeling.cluster

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    AffinityPropagation
    AgglomerativeClustering
    Birch
    BisectingKMeans
    DBSCAN
    FeatureAgglomeration
    KMeans
    MeanShift
    MiniBatchKMeans
    OPTICS
    SpectralBiclustering
    SpectralClustering
    SpectralCoclustering


snowflake.ml.modeling.compose
---------------------------------

.. currentmodule:: snowflake.ml.modeling.compose

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    ColumnTransformer
    TransformedTargetRegressor


snowflake.ml.modeling.covariance
---------------------------------

.. currentmodule:: snowflake.ml.modeling.covariance

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    EllipticEnvelope
    EmpiricalCovariance
    GraphicalLasso
    GraphicalLassoCV
    LedoitWolf
    MinCovDet
    OAS
    ShrunkCovariance


snowflake.ml.modeling.decomposition
-----------------------------------

.. currentmodule:: snowflake.ml.modeling.decomposition

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    DictionaryLearning
    FactorAnalysis
    FastICA
    IncrementalPCA
    KernelPCA
    MiniBatchDictionaryLearning
    MiniBatchSparsePCA
    PCA
    SparsePCA
    TruncatedSVD


snowflake.ml.modeling.discriminant_analysis
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.discriminant_analysis

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    LinearDiscriminantAnalysis
    QuadraticDiscriminantAnalysis


snowflake.ml.modeling.ensemble
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.ensemble

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    AdaBoostClassifier
    AdaBoostRegressor
    BaggingClassifier
    BaggingRegressor
    ExtraTreesClassifier
    ExtraTreesRegressor
    GradientBoostingClassifier
    GradientBoostingRegressor
    HistGradientBoostingClassifier
    HistGradientBoostingRegressor
    IsolationForest
    RandomForestClassifier
    RandomForestRegressor
    StackingRegressor
    VotingClassifier
    VotingRegressor


snowflake.ml.modeling.feature_selection
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.feature_selection

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    GenericUnivariateSelect
    SelectFdr
    SelectFpr
    SelectFwe
    SelectKBest
    SelectPercentile
    SequentialFeatureSelector
    VarianceThreshold


snowflake.ml.modeling.gaussian_process
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.gaussian_process

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    GaussianProcessClassifier
    GaussianProcessRegressor


snowflake.ml.modeling.impute
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.impute

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    IterativeImputer
    KNNImputer
    MissingIndicator
    SimpleImputer


snowflake.ml.modeling.kernel_approximation
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.kernel_approximation

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    AdditiveChi2Sampler
    Nystroem
    PolynomialCountSketch
    RBFSampler
    SkewedChi2Sampler


snowflake.ml.modeling.kernel_ridge
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.kernel_ridge

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    KernelRidge


snowflake.ml.modeling.lightgbm
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.lightgbm

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    LGBMClassifier
    LGBMRegressor


snowflake.ml.modeling.linear_model
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.linear_model

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    ARDRegression
    BayesianRidge
    ElasticNet
    ElasticNetCV
    GammaRegressor
    HuberRegressor
    Lars
    LarsCV
    Lasso
    LassoCV
    LassoLars
    LassoLarsCV
    LassoLarsIC
    LinearRegression
    LogisticRegression
    LogisticRegressionCV
    MultiTaskElasticNet
    MultiTaskElasticNetCV
    MultiTaskLasso
    MultiTaskLassoCV
    OrthogonalMatchingPursuit
    PassiveAggressiveClassifier
    PassiveAggressiveRegressor
    Perceptron
    PoissonRegressor
    RANSACRegressor
    Ridge
    RidgeClassifier
    RidgeClassifierCV
    RidgeCV
    SGDClassifier
    SGDOneClassSVM
    SGDRegressor
    TheilSenRegressor
    TweedieRegressor


snowflake.ml.modeling.manifold
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.manifold

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    Isomap
    MDS
    SpectralEmbedding
    TSNE


snowflake.ml.modeling.metrics
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.metrics

.. rubric:: Functions

.. autosummary::
    :toctree: api/modeling

    accuracy_score
    confusion_matrix
    correlation
    covariance
    d2_absolute_error_score
    d2_pinball_score
    explained_variance_score
    f1_score
    fbeta_score
    log_loss
    mean_absolute_error
    mean_absolute_percentage_error
    mean_squared_error
    precision_recall_curve
    precision_recall_fscore_support
    precision_score
    r2_score
    recall_score
    roc_auc_score
    roc_curve


snowflake.ml.modeling.mixture
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.mixture

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    BayesianGaussianMixture
    GaussianMixture


snowflake.ml.modeling.model_selection
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.model_selection

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    GridSearchCV
    RandomizedSearchCV


snowflake.ml.modeling.multiclass
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.multiclass

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

    OneVsOneClassifier
    OneVsRestClassifier
    OutputCodeClassifier


snowflake.ml.modeling.naive_bayes
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.naive_bayes

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

  BernoulliNB
  CategoricalNB
  ComplementNB
  GaussianNB
  MultinomialNB


snowflake.ml.modeling.neighbors
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.neighbors

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

  KernelDensity
  KNeighborsClassifier
  KNeighborsRegressor
  LocalOutlierFactor
  NearestCentroid
  NearestNeighbors
  NeighborhoodComponentsAnalysis
  RadiusNeighborsClassifier
  RadiusNeighborsRegressor


snowflake.ml.modeling.neural_network
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.neural_network

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

  BernoulliRBM
  MLPClassifier
  MLPRegressor

snowflake.ml.modeling.pipeline
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.pipeline

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

  Pipeline

snowflake.ml.modeling.preprocessing
-----------------------------------------------

.. currentmodule:: snowflake.ml.modeling.preprocessing

.. autosummary::
    :toctree: api/modeling

    StandardScaler
    OrdinalEncoder
    MinMaxScaler
    LabelEncoder
    RobustScaler
    KBinsDiscretizer
    MaxAbsScaler
    Normalizer
    OneHotEncoder
    Binarizer
    PolynomialFeatures


snowflake.ml.modeling.semi_supervised
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.semi_supervised

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

  LabelPropagation
  LabelSpreading


snowflake.ml.modeling.svm
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.svm

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

  LinearSVC
  LinearSVR
  NuSVC
  NuSVR
  SVC
  SVR



snowflake.ml.modeling.tree
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.tree

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

  DecisionTreeClassifier
  DecisionTreeRegressor
  ExtraTreeClassifier
  ExtraTreeRegressor


snowflake.ml.modeling.xgboost
-------------------------------------------

.. currentmodule:: snowflake.ml.modeling.xgboost

.. rubric:: Classes

.. autosummary::
    :toctree: api/modeling

  XGBClassifier
  XGBRegressor
  XGBRFClassifier
  XGBRFRegressor

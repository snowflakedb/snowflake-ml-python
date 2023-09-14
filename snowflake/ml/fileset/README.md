# Snowflake Python FileSet

Snowflake Python FileSet library is one of the Snowflake ML tools. It provides an easy way to unload and manage large
static data files in Snowflake's internal stage. These files then can be accessed and used in training job from
anywhere. It includes two components: Snowflake File system and Snowflake FileSet.

## Getting started

### Configure Snowflake credentials

`snowflake.ml.utils.connection_params` provides `SnowflakeLoginOptions()` function to help load Snowflake credentials
into Snowflake Python connector or Snowpark session. It recognizes the local SnowSQL configuration file and converts
the config into a dict of parameters. Follow the [guide](
    https://docs.snowflake.com/en/user-guide/snowsql-start.html#configuring-default-connection-settings) to get your
SnowSQL config file ready for the following steps.

### Setup Snowflake connection

**Prerequisite: Finish [Configure Snowflake credentials](#configure-snowflake-credentials) first.**

Snowflake filesystem requires either a
[Snowflake Python connection](https://docs.snowflake.com/en/user-guide/python-connector.html) or
[Snowpark session](https://docs.snowflake.com/en/developer-guide/snowpark/python/index.html).

To start a Snowflake Python connection:

```Python
import snowflake.connector
from snowflake.ml.utils import connection_params
conn = snowflake.connector.connect(
    **connection_params.SnowflakeLoginOptions())
```

To start a Snowpark session:

```Python
import snowflake.snowpark
from snowflake.ml.utils import connection_params
session = snowflake.snowpark.Session.builder.configs(
    connection_params.SnowflakeLoginOptions()
).create()
```

## Snowflake Filesystem APIs

Snowflake Filesystem is a Python library based on [fsspec](https://filesystem-spec.readthedocs.io/en/latest/). It
grant users read-only access to Snowflake stages as a fsspec file system.

### Create a new Snowflake Filesystem object

The Snowflake filesystem object can be created by either a Snowflake connection or Snowpark session:

```Python
from snowflake.ml.fileset import sfcfs
# Create a Snowflake filesystem with a Snowflake connection
sffs = sfcfs.SFFileSystem(sf_connection=conn)

# Create a Snowflake filesystem with a Snowpark session
sffs = sfcfs.SFFileSystem(snowpark_session=session)

# You can also create a Snowflake filesystem with fsspec interface
sffs = fsspec.filesystem("sfc", sf_connection=conn)

# The Snowflake filesystem also inherits features of fsspec. For example, it can utilize local cache:
sffs = fsspec.filesystem(
    "filecache",
    target_protocol="sfc",
    target_options={"sf_connection": sf_connection, "cache_types": "bytes", "block_size": 32 * 2**20},
    cache_storage=local_cache_path,
)
```

### List files in a stage

The Snowflake file system can list stage files under a directory in the format of
`@<database>.<schema>.<stage>/<filepath>`. Suppose we have a stage "FOO" in "MYDB" database with "public" schema. We can
 list objects inside the stage like the following:

```Python
print(sffs.ls("@MYDB.public.FOO/"))
print(sffs.ls("@MYDB.public.FOO/nytrain"))
```

It will print out files directly under the `FOO` stage and `nytrain` directory accordingly:

```python
['@MYDB.public.FOO/nytrain/']
['@MYDB.public.FOO/nytrain/data_0_0_0.csv', '@MYDB.public.FOO/nytrain/data_0_0_1.csv']
```

### Open a stage file

You can open a stage file in read mode:

```Python
with sffs.open('@MYDB.public.FOO/nytrain/nytrain/data_0_0_1.csv', mode='rb') as f:
    print(f.readline())
```

It will read the file as it was in your local file system:

```python
b'2014-02-05 14:35:00.00000054,13,2014-02-05 14:35:00 UTC,-74.00688,40.73049,-74.00563,40.70676,2\n'
```

Reading a file can also be done by using fsspec interface:

```Python
with fsspec.open("sfc://@MYDB.public.FOO/nytrain/data_0_0_1.csv", mode='rb', sf_connection=conn) as f:
    print(f.readline())
    # b'2014-02-05 14:35:00.00000054,13,2014-02-05 14:35:00 UTC,-74.00688,40.73049,-74.00563,40.70676,2\n'
```

### Other supported methods

Snowflake filesystem supports most read-only methods supported by fsspec. It includes `find()`, `info()`,
`isdir()`, `isfile()`, `exists()`, and so on.

```Python
sffs.find("@MYDB.public.FOO/")
# ['@MYDB.public.FOO/nytrain/data_0_0_0.csv', '@MYDB.public.FOO/nytrain/data_0_0_1.csv']

sffs.find("@MYDB.public.FOO/nytrain/")
# ['@MYDB.public.FOO/nytrain/data_0_0_0.csv', '@MYDB.public.FOO/nytrain/data_0_0_1.csv']

sffs.info("@MYDB.public.FOO/nytrain/data_0_0_0.csv")
# {"name": "@MYDB.public.FOO/nytrain/data_0_0_0.csv", "type": "file", size: 10}

sffs.exists("@MYDB.public.FOO/nytrain/data_0_0_0.csv")
# True
sffs.exists("@MYDB.public.FOO/nytrain/data_1_0_0.csv")
# False

sffs.isdir("@MYDB.public.FOO/nytrain/data_0_0_0.csv")
# False
sffs.isdir("@MYDB.public.FOO/nytrain/")
# True

sffs.isfile("@MYDB.public.FOO/nytrain/data_0_0_0.csv")
# False
sffs.isfile("@MYDB.public.FOO/nytrain/")
# True
```

## Snowflake FileSet APIs

A Snowflake FileSet represents an immutable snapshot of the result of a SQL query in the form of files. It is built to make
user's life easier when do machine learning tasks.

### Create a new FileSet object

FileSet object can be created with either a Snowflake Python connection, or a Snowpark dataframe. It also needs a Snowflake
stage path as one of the inputs. A fully qualified stage will be a Snowflake internal stage with server side encryption.
The stage path should be represented as `@<database>.<schame>.<stage>/<optional_subdirectories>`

#### New FileSet with a Snowflake Python connection & a SQL query

```Python

train_fileset = fileset.FileSet.make(
    target_stage_loc=fully_qualified_stage,
    name="train",
    sf_connection=conn,
    query="SELECT * FROM Mytable limit 1000000",
)
```

#### New FileSet with a Snowpark dataframe

```Python
df = session.sql("SELECT * FROM Mytable limit 1000000")

train_fileset = fileset.FileSet.make(
    target_stage_loc=fully_qualified_stage,
    name="train",
    snowpark_dataframe=train_data_df,
)
```

#### Caveat: Data type Casting

At `FileSet.make()` we cast the data type of the input query / dataframe into types that are commonly accepted by
machine learning libraries like PyTorch and TensorFlow.

##### Supported data types

For supported data types, the casting will happen implicitly. The followings are supported data types:
| Snowflake Data Type        | FileSet Casted Data Type |
| -------------------------- | ------------------------ |
| NUMBER with zero scale     | int                      |
| NUMBER with non-zero scale | float                    |
| Float/REAL                 | float                    |
| BINARY                     | binary                   |
| TEXT                       | string                   |
| BOOLEAN                    | boolean                  |

##### Unsupported data types

For unsupported data types, there will be no data casting. A warning will be logged to notify users to handle these data
 types beforehand, as these data types will not be exported to torch anyway. Unsupported snowflake data types includes:

- DATE
- TIME
- TIMESTAMP
- TIMESTAMP_LTZ
- TIMESTAMP_TZ
- TIMESTAMP_NTZ
- OBJECT
- VARIANT
- ARRAY

### Feed Pytorch

Once you created a `FileSet`, you can get a torch `DataPipe` and give it
to a torch `DataLoader`. The `DataLoader` iterates through the data in the
`FileSet` and produces batched torch Tensors.

```Python
from torch.utils.data import DataLoader

# PLEASE NOTE:
# 1. shuffle, batching are handled by the DataPipe produced by
#   `to_torch_datapipe` for performance reasons. That's why those
#   functionalities should be disabled in `DataLoader`.
# 2. `num_workers` must be set to 0. Multi-processing DataLoader is
#   not supported (and will not provide better performance even if
#   it works.)

train_dl = DataLoader(
    train_fileset.to_torch_datapipe(
        batch_size=8192, shuffle=True, drop_last_batch=True
    ),
    num_workers=0,
    batch_size=None,
)
```

### Feed TensorFlow

Similarly, you can get a `tf.data.Dataset` from a `FileSet`. Again, the `Dataset` dispenses batched TF
Tensors.

```Python
import tensorflow as tf

# Prefer to set batch_size, shuffle here instead of chaining .batch(batch_size)
# and .shuffle() for better efficiency.
ds = fileset_init_with_snowpark.to_tf_dataset(
    batch_size=4, shuffle=True, drop_last_batch=True)
assert(isinstance(ds, tf.data.Dataset))
```

### Delete FileSet

You can explicitly call `delete()` to delete the FileSet and its underlying stage. If it is not called, the stage will
be preserved, and you can recover it with the path to that stage.

To delete a FileSet, simply call `delete()`:

```Python
train_fileset = fileset.FileSet.make(...)
...
train_fileset.delete()
```

### Recover existing data

If a old FileSet is not deleted, you can recover it in another Python program with its stage path and name:

```Python
train_fileset = fileset.FileSet(
    snowpark_session=session,
    target_stage_loc=fully_qualified_stage,
    name="train",
)

```

### Retrieve all underlying files for some other use-case

Each FileSet contains a list of parquet files. You can get the list of files by

```Python
train_fileset.files()
# ['sfc://@mydb.myschema.mystage//train/data_0_0_0.snappy.parquet']
```

The returned file path can be opened with [Snowflake filesystem](#snowflake-filesystem-apis).

## Examples

### Use Snowflake File system to process Snowflake stage files

With Snowflake Filesystem, you can easily read Snowflake stage files as they are on you local file system:

```Python
from snowflake.ml.fileset import sfcfs

sffs = sfcfs.SFFileSystem(sf_connection=conn)

# The stage should be with Snowflake server-side encryption.
stage_path = "@MYDB.public.FOO/"

# List files/directories in the stage
sffs.ls(stage_path)                 # ['@MYDB.public.FOO/nytrain/', '@MYDB.public.FOO/dogs/']

# List files in a subdirectory
sffs.ls("@MYDB.public.FOO/nytrain") # ['@MYDB.public.FOO/nytrain/data_0_0_0.csv', '@MYDB.public.FOO/nytrain/data_0_0_1.csv']

# Read a stage file
with sffs.open('@MYDB.public.FOO/nytrain/nytrain/data_0_0_1.csv', mode='rb') as f:
    print(f.readline())             # b'2014-02-05 14:35:00.00000054,13,2014-02-05 14:35:00 UTC,-74.00688,40.73049,-74.00563,40.706762\n'

# Read unstructured data
with sffs.open("@MYDB.public.FOO/dogs/dog1.png", mode='rb') as f:
    ...
```

### Use FileSet to materilze and load data

A FileSet could help materialize your SQL query and feed the data into Pytorch.

```Python
import fsspec
import pyarrow.parquet as pq
from snowflake.ml.fileset import fileset

# Create a stage with Snowflake server-side encryption
# `encryption = (type = 'snowflake_sse')` is necessary since Snowflake FileSet and Filesystem only works with
#     server-side encryption.
session.sql(
    f"create stage MYDB.public.FOO encryption = (type = 'snowflake_sse')"
).collect()

# Create a Snowpark dataframe
preprocessed_train_data = session.sql(
    "SELECT * FROM Mytable limit 1000000"
)

# Create a FileSet
fully_qualified_stage = f"@MYDB.public.FOO"
train_fileset = fileset.FileSet.make(
    target_stage_loc=fully_qualified_stage,
    name="nytrain",
    snowpark_dataframe=preprocessed_train_data,
)

# List materialized parquet files in this FileSet
files = train_fileset.files() # ["sfc://@MYDB.public.FOO/nytrain/data_0_0_0.snappy.parquet", "sfc://@MYDB.public.FOO/nytrain/data_0_0_1.snappy.parquet"]

# You can manually read those underlying parquet files with pyarrow.parquet API
for file in files:
    with fsspec.open(file, mode='rb', sf_connection=conn) as f:
        pq.read_table(f)
        ...

# Feed data into Pytorch Dataloader
train_dl = DataLoader(
    train_fileset.to_torch_datapipe(
        batch_size=8192, shuffle=True, drop_last_batch=True
    ),
    num_workers=0,
    batch_size=None,
)

# Training with Pytorch Dataloader
for batch in train_dl:
    # Some algorithms ...
```

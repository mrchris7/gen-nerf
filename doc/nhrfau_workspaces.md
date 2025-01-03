# Lustre Workspace

The "NVMe Lustre (anvme)" is a workspace accessible on the Alex cluster of the [Erlangen National High Performance Computing Center (NHR@FAU)](https://hpc.fau.de/).

## Data Handling
The "Lustre workspace" is recommended to store large datasets. Note that it is not recommended to store and load many small files directly from it.

Instead, smaller files need to be organized and packed as archives. When training, accessing files (via NFS) directly from the workspace creates network traffic and can potentially slow down the access for all other users on the cluster. In practice, at the start of each job, the dataset needs to be copied and unpacked to node-local storage (```$TMPDIR```). Loading data for training from node-local storage is efficient and can be done for many small files.

Compared to the ```$WORK``` filesystem (which can also be used to store datasets), the "Lustre workspace" allows to copy data much faster to node-local storage.


## Workspace Usage

When a workspace is created the user receives a path where data can be stored. The created workspace has a limited lifetime (duration). When the duration of a workspace ends, the workspace will be deleted. The lifetime can be extended multiple times.


Create a workspace:
```
ws_allocate <name> [<days>]
```

Get the path to a workspace:

```
ws_find <name>
```

Set a new duration of a workspace:
```
ws_extend <name> [<days>]
```

Delete a workspace, before its lifetime is up:

```
ws_release <name>
```

More information can be found here: https://doc.nhr.fau.de/data/workspaces/
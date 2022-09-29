# instruction for saving graph dataset

if graph is homogenous, it has to be saved as `.mat` file which contains a dict of the following form:

```
{
    'features': COO sparse matrix of shape (num_nodes, num_features)
    'labels': ndarray of shape (1, num_nodes)
    'edges': ndarray of shape (2, num_edges)
}
```

if graph is heterogenous, it has to be saved in the same format, but the `edges` pair changes to per relation pairs

it doesn't matter if the graph contains self-loops or is directed — on processing stage, all self-loops are removed, and all edges are made bidirected (so the graph becomes undirected)
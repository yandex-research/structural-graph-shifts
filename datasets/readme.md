# Graph Datasets & Data Splits

The homogenous graph information is stored in `.mat` files using the following dict structure: 
- `'features'`: `scipy.sparse.coo_matrix` or `np.ndarray` of shape `(num_nodes, num_features)`
- `'labels'`: `np.ndarray` of shape `(1, num_nodes)`
- `'edges'`: `np.ndarray` of shape `(2, num_edges)`

Heterogenous graphs are saved in the same format, but the value for `'edges'` becomes a dict of relation names and their edges. Morevoer, it doesn't matter if the graph contains self-loops or is directed — on processing stage, all self-loops are removed, and all edges are made bidirected (so the graph becomes undirected).

If you need an access only to the content of a particular `.mat` file located at `dataset_path`, you can open it using `scipy.io.loadmat(dataset_path)`. 
The associated data splits are stored in `.pth` files in the form af dicts, with keys from `['train', 'valid_in', 'valid_out', 'test_in', 'test_out']` and values represented as `torch.tensor` of shape `(num_nodes,)`.
# Graph Datasets & Data Splits

The homogenous graph information is stored in `.mat` files using the following dict structure: 
- `'features'`: `scipy.sparse.coo_matrix` or `np.ndarray` of shape `(num_nodes, num_features)`
- `'labels'`: `np.ndarray` of shape `(1, num_nodes)`
- `'edges'`: `np.ndarray` of shape `(2, num_edges)`

Heterogenous graphs are saved in the same format, but the value for `'edges'` becomes a dict of relation names and their edges. Morevoer, it doesn't matter if the graph contains self-loops or is directed — on processing stage, all nodes are supplied with self-loops and all edges are made bidirected.

If you need an access only to the content of a particular `.mat` file located at `dataset_path`, you can open it using `scipy.io.loadmat(dataset_path)`. 
The associated data splits are stored in `.pth` files in the form af dicts, with keys from `['train', 'valid_in', 'valid_out', 'test_in', 'test_out']` and values represented as `torch.tensor` of shape `(num_nodes,)`.

To prepare one of the proposed splits on some of the considered graph datasets, it is necessary to make a proper config file `config.yaml`. There, you should specify 
- `dataset_name`: name of graph dataset (e.g., `amazon-computer`)
- `strategy_name`: name of split strategy (e.g., `popularity`)
- `in_train_size`: size of ID train part
- `in_valid_size`: size of ID valid part
- `in_test_size`: size of ID test part
- `out_valid_size`: size of OOD valid part
- `out_test_size`: size of OOD test part

The introduced part sizes must sum to `1.0`. To apply a split strategy different from those proposed in our paper, you should implement a function to compute the corresponding node-level graph property (see `utils.py` for the existing implementations of `popularity`, `locality` and `density` splits).
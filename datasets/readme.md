# Graph Datasets & Data Splits

The homogenous graph information is stored in `.mat` files using the following dict structure: 
- `'features'`: `scipy.sparse.coo_matrix` or `np.ndarray` of shape `(num_nodes, num_features)`
- `'labels'`: `np.ndarray` of shape `(1, num_nodes)`
- `'edges'`: `np.ndarray` of shape `(2, num_edges)`

Heterogenous graphs are saved in the same format, but the value for `'edges'` becomes a dict of relation names and their edges. Morevoer, it doesn't matter if the graph contains self-loops or is directed — on processing stage, all self-loops are removed, and all edges are made bidirected (so the graph becomes undirected).

If you need an access only to the content of a particular `.mat` file located at `dataset_path`, you can open it using `scipy.io.loadmat(dataset_path)`. 
The associated data splits are stored in `.pth` files in the form af dicts, with keys from `['train', 'valid_in', 'valid_out', 'test_in', 'test_out']` and values represented as `torch.tensor` of shape `(num_nodes,)`.

To prepare one of the proposed splits on some of the considered graph datasets, it is necessary to make a proper config file `config.yaml`. There, you should specify 
- `dataset_name`: name of graph dataset (e.g., `pubmed`)
- `strategy_name`: name of split strategy (e.g., `pagerank`)
- `version_name`: optinal version name of data split (e.g., `extra`)
- `in_size`: size of ID subset
- `out_size`: size of OOD subset
- `in_train_size`: size of train part
- `in_valid_size`: size of ID valid part
- `out_valid_size`: size of OOD valid part

Please take into account that sizes should be described by their absolute values, not relative to the containing subset (e.g., if `in_size` is 0.5 and `in_train_size` is half of ID subset, you should specify 0.25 for it). Note that `in_test_size` and `out_test_size` are inferred from the specified values.

To apply a split strategy different from those proposed in our paper, you should implement a function to compute the corresponding node-level graph property (see `utils.py` for the existing implementations of `pagerank`, `personalised` and `clustering` splits).
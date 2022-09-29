from scipy import io

import torch
import dgl


relations = {
    'fraud-amazon': ['upu', 'usu', 'uvu']
}

node_names = {
    'fraud-amazon': 'user'
}


### prepare functions


def prepare_homogenous_graph(dataset_root, dataset_name, prepare_config):
    dataset_path = f"{dataset_root}/{dataset_name}/data.mat"
    data = io.loadmat(dataset_path)

    features = data['features'].todense()
    labels = data['labels'].squeeze()

    rows, cols = data['edges']
    edge_index = torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)
    
    graph = dgl.graph(edge_index, num_nodes=len(features))
    graph.ndata['features'] = torch.tensor(features, dtype=torch.float)
    graph.ndata['labels'] = torch.tensor(labels, dtype=torch.long)
    
    return graph


def prepare_heterogenous_graph(dataset_root, dataset_name, prepare_config):
    dataset_path = f"{dataset_root}/{dataset_name}/data.mat"
    data = io.loadmat(dataset_path)

    features = data['features'].todense()
    labels = data['labels'].squeeze()
        
    edge_data = {}
    for relation in relations[dataset_name]:

        if relation not in prepare_config['relations']:
            continue
        
        rows, cols = data[relation]
        edge_index = torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)

        edge_type = (node_names[dataset_name], relation, node_names[dataset_name])
        edge_data[edge_type] = edge_index

    graph = dgl.heterograph(edge_data)
    graph.ndata['features'] = torch.tensor(features, dtype=torch.float)
    graph.ndata['labels'] = torch.tensor(labels, dtype=torch.long)
    
    return graph


dataset_name_to_prepare_graph_fn = {
    'fraud-amazon': prepare_heterogenous_graph,

    'amazon-computer': prepare_homogenous_graph,
    'amazon-photo': prepare_homogenous_graph,
    
    'cora-ml': prepare_homogenous_graph,
    'citeseer': prepare_homogenous_graph,
    'pubmed': prepare_homogenous_graph,

    'coauthor-cs': prepare_homogenous_graph,
    'coauthor-physics': prepare_homogenous_graph,
}


### process functions


def process_homogenous_graph(graph, process_config):
    # make graph undirected
    graph_undirected = dgl.to_bidirected(graph, copy_ndata=True)

    # remove self loops to avoid problems with message passing
    return dgl.remove_self_loop(graph_undirected)


def process_heterogenous_graph(graph, process_config):
    # TODO: gonna need only a couple of relations in the future
    
    # merge all types of edges, prepare homogeneous graph
    graph_homogeneous = dgl.to_homogeneous(graph, ndata=list(graph.ndata.keys()), edata=None, store_type=False)

    # remove multi-edges, prepare completely bidirected graph
    graph_simple = dgl.to_simple(graph_homogeneous)

    return process_homogenous_graph(graph_simple, process_config)


dataset_name_to_process_graph_fn = {
    'fraud-amazon': process_heterogenous_graph,

    'amazon-computer': process_homogenous_graph,
    'amazon-photo': process_homogenous_graph,
    
    'cora-ml': process_homogenous_graph,
    'citeseer': process_homogenous_graph,
    'pubmed': process_homogenous_graph,

    'coauthor-cs': process_homogenous_graph,
    'coauthor-physics': process_homogenous_graph,
}


### masks functions


def prepare_homogenous_masks(dataset_root, dataset_name, split_config):
    split_path = f"{dataset_root}/{dataset_name}/splits/type={split_config['type']}.pth"
    masks = torch.load(split_path)
    return masks


def prepare_heterogenous_masks(dataset_root, dataset_name, split_config):
    split_path = f"{dataset_root}/{dataset_name}/splits/relation={split_config['relation']}_type={split_config['type']}.pth"
    masks = torch.load(split_path)
    return masks


dataset_name_to_prepare_masks_fn = {
    'fraud-amazon': prepare_heterogenous_masks,

    'amazon-computer': prepare_homogenous_masks,
    'amazon-photo': prepare_homogenous_masks,
    
    'cora-ml': prepare_homogenous_masks,
    'citeseer': prepare_homogenous_masks,
    'pubmed': prepare_homogenous_masks,

    'coauthor-cs': prepare_homogenous_masks,
    'coauthor-physics': prepare_homogenous_masks,
}
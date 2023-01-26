import os
import shutil
import argparse

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split_config_path', 
        type=str,
        help='configuration file for preparing data splits'
    )

    options = parser.parse_args()
    split_config = utils.prepare_split_config(options.split_config_path)

    graph_dgl = utils.prepare_dgl_graph(split_config)
    graph_gt = utils.prepare_graphtool_graph(graph_dgl)

    split_params = utils.prepare_split_params(split_config)
    split_masks, property_values = utils.prepare_split_masks(graph_gt, split_params, split_config)
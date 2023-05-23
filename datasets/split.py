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
    part_ratios = utils.prepare_part_ratios(split_config)
    graph_dgl = utils.prepare_graph(split_config)

    split_masks, property_values = utils.prepare_split(graph_dgl, part_ratios, split_config)
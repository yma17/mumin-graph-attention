"""Script for claim classification task."""

import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))
# Add paths to all model scripts
# TODO: add GS path

hgs_path = os.path.join(curr_path, '../mumin-baseline/src/')
sys.path.append(hgs_path)

# TODO: add HAN path

magnn_path = os.path.join(curr_path, '../MAGNN/')
sys.path.append(magnn_path)

import argparse
from train_graph_model import train_graph_model

import logging

# Set up logging
fmt = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


def claim_classification(task, model, size):
    assert task in ["claim", "tweet"]
    assert model in ["gs", "hgs", "han", "magnn"]
    assert size in ["small", "medium", "large"]

    if model == "gs":
        pass  # TODO
    elif model == "hgs":
        scores = train_graph_model(task=task, size=size)
    elif model == "han":
        pass  # TODO
    elif model == "magnn":
        pass  # TODO

    # Report statistics
    log = 'Final evaluation\n'
    for split, dct in scores.items():
        for statistic, value in dct.items():
            statistic = split + '_' + statistic.replace('eval_', '')
            log += f'> {statistic}: {value}\n'
    logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')  # required argument
    parser.add_argument('--model')  # required argument
    parser.add_argument('--size')  # required argument
    args = parser.parse_args()

    claim_classification(args.model, args.size)
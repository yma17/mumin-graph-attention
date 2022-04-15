"""Script for claim classification task."""

import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(curr_path, '../../mumin-baseline/src/')
print(model_path)
sys.path.append(model_path)

import argparse
from train_graph_model import train_graph_model

import logging

# Set up logging
fmt = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


def claim_classification(model, size):
    assert model in ["hgc", "han"]
    assert size in ["small", "medium", "large"]

    if model == "hgc":
        scores = train_graph_model(task='claim', size=size)
    else:  # model == "han"
        pass  # TODO

    # Report statistics
    log = 'Final evaluation\n'
    for split, dct in scores.items():
        for statistic, value in dct.items():
            statistic = split + '_' + statistic.replace('eval_', '')
            log += f'> {statistic}: {value}\n'
    logger.info(log)

    # Write results to file
    # TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')  # required argument
    parser.add_argument('--size')  # required argument
    args = parser.parse_args()

    claim_classification(args.model, args.size)
"""AutoML.

Runs the BOHB search

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

References:
    <https://automl.github.io/HpBandSter/build/html/auto_examples/example_1_
     local_sequential.html>
"""
import argparse
import datetime
import os
import pickle
import traceback

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from . import SearchWorker

import logging

logging.basicConfig(level=logging.WARNING)


def parse_args():
    """Parses command line arguments."""
    description = "runs a BOHB search for hyperparameters"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data_path', metavar='D', type=str,
                        help='path to the data folder')
    parser.add_argument('output_dir', metavar='O', type=str,
                        help='directory for the result output')
    parser.add_argument('min_budget', metavar='L', type=int,
                        help='minimum budget to use during optimization')
    parser.add_argument('max_budget', metavar='M', type=int,
                        help='maximum budget to use during optimization')
    parser.add_argument('iterations', metavar='I', type=int,
                        help='number of iterations to perform')

    return parser.parse_args()


def run_optimization(args):
    """Runs the optimization process."""
    print("Starting name server.")
    date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')

    # First start nameserver
    NS = hpns.NameServer(run_id=date_time, host='127.0.0.1', port=None)
    NS.start()

    print("Preparing result logger and loading previous run, if it exists.")

    # Also start result logger
    output_dir = os.path.join(args.output_dir,
                              datetime.datetime.now()
                              .strftime('%Y_%m_%d--%H_%M_%S'))
    result_logger_path = os.path.join(output_dir, 'results_log.json')
    best_result_path = os.path.join(output_dir, 'best_config.txt')

    print("Result logger will be written to %s" % result_logger_path)
    if os.path.exists(result_logger_path):
        previous_run = hpres.logged_results_to_HBS_result(result_logger_path)
    else:
        previous_run = None

    result_logger = hpres.json_result_logger(directory=output_dir,
                                             overwrite=True)

    print("Starting search worker.\n")

    # Then start worker
    w = SearchWorker(args.data_path, os.path.join(output_dir, "logging"),
                     nameserver='127.0.0.1', run_id=date_time)
    w.run(background=True)

    print("Initializing optimizer.")
    # Run the optimizer
    bohb = BOHB(configspace=w.get_configspace(),
                run_id=date_time,
                nameserver='127.0.0.1',
                result_logger=result_logger,
                min_budget=args.min_budget,
                max_budget=args.max_budget,
                previous_result=previous_run)

    print("Initialization complete. Starting optimization run.")

    res = bohb.run(n_iterations=args.iterations)

    print("Optimization complete.")
    output_fp = os.path.join(output_dir, 'results.pkl')

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print("Results will be saved at:\n{}".format(output_fp))
    print("Best found configuration: ", id2config[incumbent]['config'])

    with open(best_result_path, mode='w') as file:
        lines = ["Best results are as follows:\n",
                 "{}".format(id2config[incumbent]['config'])]
        file.writelines(lines)

    with open(output_fp, mode='wb') as file:
        pickle.dump(res, file)

    # Shutdown after completion
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()


if __name__ == '__main__':
    args = parse_args()
    output_dir = os.path.join(args.output_dir,
                              datetime.datetime.now().
                              strftime('%Y_%m_%d--%H_%M_%S'))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    try:
        lines = ["Starting optimization run with the following parameters:\n",
                 "    Data path:      {}\n".format(args.data_path),
                 "    Output dir:     {}\n".format(output_dir),
                 "    Minimum budget: {}\n".format(args.min_budget),
                 "    Maximum budget: {}\n".format(args.max_budget),
                 "    Iterations:     {}\n".format(args.iterations)]
        with open(os.path.join(output_dir, 'optimizer_configuration.txt'),
                  'w') as file:
            # Write configuration to file so we remember what happened
            file.writelines(lines)

        for line in lines:
            # Print out of the current configuration
            print(line, end='')

        run_optimization(args)
    finally:
        exception_encountered = traceback.format_exc(0)
        if "SystemExit" in exception_encountered \
                or "KeyboardInterrupt" in exception_encountered \
                or "None" in exception_encountered:
            pass

        else:
            print("I Died")
            with open(os.path.join(os.getcwd(), "traceback.txt"), mode="w") \
                    as file:
                traceback.print_exc(file=file)
        pass

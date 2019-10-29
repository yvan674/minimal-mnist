"""Explorer.

Explore the results from the hyperparameter optimization process.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import json


def load_json_as_lists():
    """Returns json as a single list with name, results, config."""
    results_dir = '/Users/Yvan/Documents/Projects/Physical NN/Hpoptim Results/'
    results_dir += '2019_10_23--12_26_11/'

    results_list = []
    results_file = open(results_dir + 'results.json')
    config_file = open(results_dir + 'configs.json')

    read_results = []
    for line in results_file:
        read_results.append(json.loads(line))

    read_config = []
    for line in config_file:
        read_config.append(json.loads(line))

    results_file.close()
    config_file.close()

    read_config_dict = dict()
    # Turn config into dict
    for line in read_config:
        read_config_dict[str(line[0])] = line[1]

    for i in range(len(read_results)):
        if read_results[i][1] > 23:
            run_id = read_results[i][0]
            results_list.append({
                'run_number': i,
                'iter_number': read_results[i][0],
                'validation_acc': read_results[i][3]['info']['validation accuracy'],
                'config': read_config_dict[str(run_id)]
            })

    return results_list

def filter_below(threshold, results):
    """Filters items below a certain threshold out."""
    out_list = []
    for line in results:
        if line['validation_acc'] > threshold:
            out_list.append(line)

    return out_list


def sort_by_node_usage(results):
    """Sorts items by amount of nodes used."""
    out_list = []
    for line in results:
        if line['config']['first_layer'] + line['config']['second_layer'] < 65:
            out_list.append(line)

    return out_list


if __name__ == '__main__':
    results = load_json_as_lists()

    results = filter_below(0.90, results)
    results = sort_by_node_usage(results)

    print(len(results))
    input()
    for line in results:
        print(line)
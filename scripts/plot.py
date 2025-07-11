import pandas as pd
import seaborn as sns

from argparse import ArgumentParser
from matplotlib import pyplot as plt


def parse_args():
    parser = ArgumentParser(description='Plot the result of the solver')
    parser.add_argument('result_file', type=str, help='result file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    res = pd.read_csv(args.result_file)
    res = res[res.result != 'skipped']
    # sort by runtime
    res = res.sort_values(by='running time')
    res = res.reset_index(drop=True)
    res['index'] = res.index
    print(res)
    sns.scatterplot(data=res, x='index', y='running time')
    plt.savefig('result.png')

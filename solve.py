import json
import argparse
import logging
import logzero
import pandas as pd
import pickle

from wfomc import Algo
from logzero import logger
from contexttimer import Timer

from cofola.parser.parser import parse
from cofola.solver import solve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, type=str)
    parser.add_argument('--ids', '-ids', type=str, nargs='+',
                        help='List of problem ids to solve, '
                        'None means all problems in the input file')
    parser.add_argument('--debug', '-d', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    input_file = args.input
    all_problems = json.load(open(input_file, 'r'))
    ids = []
    df = []
    if args.ids is None:
        ids = all_problems.keys()
    else:
        for i in args.ids:
            if '-' in i:
                start, end = i.split('-')
                ids.extend(str(j) for j in range(int(start), int(end) + 1))
            else:
                ids.append(i)
    wfomc_problems = dict()
    for i in ids:
        problem = all_problems[i]
        tags = problem['tags']
        unchecked = [
            'timeout', 'unencodeable'
        ]
        checked = [
            'circle', 'sequence'
        ]
        if any(tag in tags for tag in unchecked) \
                or not any(tag in tags for tag in checked):
            df.append({
                'problem_id': i,
                'gt_result': problem['answer'],
                'result': 'skipped',
                'running time': 0,
                'unencodeable_reason': ', '.join(tags)
            })
        else:
            gt_result = problem['answer']
            with Timer() as t:
                try:
                    cofola_problem = parse(problem['program'])
                    res, problems, full_circle = solve(cofola_problem, Algo.FASTv2,
                                use_partition_constraint=True,
                                lifted=False)
                    if full_circle:
                        wfomc_problems[i] = problems
                except Exception as e:
                    res = 'error'
                    logger.exception(e)
            if res != int(gt_result):
                logger.error(f'The answer is wrong for problem {i}: {res} != {gt_result}')
                # exit(1)
            df.append({
                'problem_id': i,
                'gt_result': gt_result,
                'result': res,
                'running time': t.elapsed,
                'unencodeable_reason': ''
            })
        pd_df = pd.DataFrame(df)
        pd_df.to_csv('results.csv', index=False)
    with open('wfomc_problems.pkl', 'wb') as f:
        pickle.dump(wfomc_problems, f)
    print(pd_df)

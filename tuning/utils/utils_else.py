import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

def normalize_path(path):
    return Path(path).resolve()

def get_cmd(sys_argv):
    arguments = sys_argv[1:]
    arguments_string = ' '.join(arguments)
    return arguments_string


def replace_spaces_with_underscores(input_string):
    return input_string.replace(' ', '_')

# nodelist cn[0-32]
def get_file_name(input_string, nodelist):
    parts = input_string.split()
    for i, part in enumerate(parts):
        # change filename to print log
        if part == "-hosts" and i + 1 < len(parts):
            parts[i + 1] = nodelist
    return str(" ".join(parts))

def search_file(directory, filename):
    print(os.path.join(directory, filename))
    return os.path.exists(os.path.join(directory, filename))

def get_cwd():
    return os.getcwd()


def convert_feature(df):
    df['agg_perf_by_slowest'] = (df['POSIX_BYTES_READ'] + df['POSIX_BYTES_WRITTEN']) / (
            df['POSIX_F_READ_TIME'] + df['POSIX_F_WRITE_TIME'] + df['POSIX_F_META_TIME'])
    df['POSIX_WRITE_PER_OPEN'] = df['POSIX_WRITES'] / df['POSIX_OPENS']
    df['POSIX_READ_PER_OPEN'] = df['POSIX_READS'] / df['POSIX_OPENS']

    for index, row in df.iterrows():
        if row['POSIX_WRITES'] != 0:
            df.loc[index, 'POSIX_FSYNC_PER_WRITE'] = row['POSIX_FSYNCS'] / row['POSIX_WRITES']
        else:
            df.loc[index, 'POSIX_FSYNC_PER_WRITE'] = 0

    log_F = ['POSIX_OPENS', 'POSIX_FSYNCS', 'POSIX_WRITE_PER_OPEN', 'POSIX_READ_PER_OPEN', 'POSIX_FSYNC_PER_WRITE',
             'NPROCS',
             'POSIX_READS', 'POSIX_WRITES',
             'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES', 'POSIX_CONSEC_READS',
             'POSIX_CONSEC_WRITES',
             'POSIX_ACCESS1_ACCESS', 'POSIX_ACCESS1_COUNT',
             'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN',
             'MPIIO_COLL_READS', 'MPIIO_COLL_WRITES']

    add_small_value = 0.1
    set_NaNs_to = -10
    for c in log_F:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(value=set_NaNs_to)
        df[c + "_LOG10"] = np.log10(df[c] + add_small_value).fillna(value=set_NaNs_to)
    return df


def get_common_prefix(dictionary):
    keys = list(dictionary.keys())
    if not keys:
        return ""

    common_prefix = keys[0]
    for key in keys[1:]:
        i = 0
        while i < len(common_prefix) and i < len(key) and common_prefix[i] == key[i]:
            i += 1
        common_prefix = common_prefix[:i]

    return common_prefix


def get_last_slash_prefix(string):
    last_slash_index = string.rfind("/")
    if last_slash_index == -1:
        return ""
    else:
        return string[:last_slash_index + 1]


def get_last_slash_postfix(string):
    last_slash_index = string.rfind("/")
    if last_slash_index == -1:
        return string  # 如果路径中没有斜杠，返回整个路径
    else:
        return string[last_slash_index + 1:]


def kill_proot():
    ps_output = subprocess.check_output(['ps', 'aux']).decode('utf-8')
    proot_processes = [line.split()[1] for line in ps_output.splitlines() if 'proot' in line and 'grep' not in line]
    if proot_processes:
        print(f"Terminating proot process (PID: {proot_processes[0]})...")
        subprocess.run(['kill', proot_processes[0]])
        print("Proot process terminated.")
    else:
        print("No proot process found.")


def huber_approx_obj(y_pred, y_test):
    """
    Huber loss, adapted from https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
    """
    d = y_pred - y_test
    h = 5  # h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess
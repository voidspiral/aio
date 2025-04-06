import math
import re
import logging
import subprocess
import sys
import numpy as np
import pandas as pd
import glob

# 本地定义特征列表
features18_ = ['POSIX_OPENS_LOG10', 'POSIX_SEEKS_LOG10', 'POSIX_STATS_LOG10', 'POSIX_MMAPS_LOG10', 'POSIX_FSYNCS_LOG10',
               'POSIX_MODE_LOG10', 'POSIX_MEM_ALIGNMENT_LOG10', 'POSIX_FILE_ALIGNMENT_LOG10', 'NPROCS_LOG10',
               'POSIX_TOTAL_ACCESSES_LOG10', 'POSIX_TOTAL_BYTES_LOG10', 'POSIX_TOTAL_FILES_LOG10', 'NODES',
               'POSIX_BYTES_READ_PERC', 'POSIX_UNIQUE_BYTES_PERC', 'POSIX_SHARED_BYTES_PERC',
               'POSIX_READ_ONLY_BYTES_PERC',
               'POSIX_READ_WRITE_BYTES_PERC']

log_features = ['POSIX_OPENS', 'POSIX_SEEKS', 'POSIX_STATS', 'POSIX_MMAPS', 'POSIX_FSYNCS', 'POSIX_MODE',
                'POSIX_MEM_ALIGNMENT', 'POSIX_FILE_ALIGNMENT', 'NPROCS', 'POSIX_TOTAL_ACCESSES', 'POSIX_TOTAL_BYTES',
                'POSIX_TOTAL_FILES', 'NODES']

perc_features1 = ['POSIX_BYTES_READ', 'POSIX_UNIQUE_BYTES', 'POSIX_SHARED_BYTES', 'POSIX_READ_ONLY_BYTES',
                  'POSIX_READ_WRITE_BYTES', 'POSIX_WRITE_ONLY_BYTES', 'POSIX_UNIQUE_FILES', 'POSIX_SHARED_FILES',
                  'POSIX_READ_ONLY_FILES', 'POSIX_READ_WRITE_FILES', 'POSIX_WRITE_ONLY_FILES', 'POSIX_READS',
                  'POSIX_WRITES', 'POSIX_RW_SWITCHES', 'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES', 'POSIX_CONSEC_READS',
                  'POSIX_CONSEC_WRITES', 'POSIX_FILE_NOT_ALIGNED', 'POSIX_MEM_NOT_ALIGNED', 'POSIX_ACCESS1_COUNT',
                  'POSIX_ACCESS2_COUNT', 'POSIX_ACCESS3_COUNT', 'POSIX_ACCESS4_COUNT']

perc_features2 = ['POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K', 'POSIX_SIZE_READ_1K_10K',
                  'POSIX_SIZE_READ_10K_100K',
                  'POSIX_SIZE_READ_100K_1M', 'POSIX_SIZE_READ_1M_4M', 'POSIX_SIZE_READ_4M_10M',
                  'POSIX_SIZE_READ_10M_100M',
                  'POSIX_SIZE_READ_100M_1G', 'POSIX_SIZE_READ_1G_PLUS', 'POSIX_SIZE_WRITE_0_100',
                  'POSIX_SIZE_WRITE_100_1K',
                  'POSIX_SIZE_WRITE_1K_10K', 'POSIX_SIZE_WRITE_10K_100K', 'POSIX_SIZE_WRITE_100K_1M',
                  'POSIX_SIZE_WRITE_1M_4M',
                  'POSIX_SIZE_WRITE_4M_10M', 'POSIX_SIZE_WRITE_10M_100M', 'POSIX_SIZE_WRITE_100M_1G',
                  'POSIX_SIZE_WRITE_1G_PLUS']


def column_sum(list1, list2):
    # 使用zip()函数将两个列表按列组合
    combined = zip(list1, list2)

    # 利用列表推导式进行按列相加
    column_sums = [x + y for x, y in combined]

    # 返回按列相加的结果
    return column_sums


def get_number_columns(df):
    """
    Since some columns contain string metadata, and others contain values,
    this function returns the columns that contain values.
    """
    return df.columns[np.logical_or(df.dtypes == np.float64, df.dtypes == np.int64)]


def convert_POSIX_features_to_percentages(df, remove_dual=False):
    """
    Certain features like POSIX_SEQ_READS make more sense when normalized by a more general feature such as POSIX_READS
    For all features that measure either the number of a certain type of access, or the number of bytes, we normalize by
    the total number POSIX accesses and total number of POSIX bytes accessed.
    If remove_dual is true, removes one of the dual features such read and write percentage, unique and shared, etc.
    """
    df = df.copy()

    if np.any(np.isnan(df[get_number_columns(df)])):
        logging.error("Found NaN values before normalizing dataframe.")

    # 保存NODES列
    nodes_data = None
    if 'NODES' in df.columns:
        nodes_data = df['NODES'].copy()
        df = df.drop(columns=['NODES'])

    total_accesses = df.POSIX_TOTAL_ACCESSES
    total_bytes = df.POSIX_TOTAL_BYTES
    total_files = df.POSIX_TOTAL_FILES

    try:
        df['POSIX_BYTES_READ_PERC'] = df.POSIX_BYTES_READ / total_bytes
        df['POSIX_UNIQUE_BYTES_PERC'] = df.POSIX_UNIQUE_BYTES / total_bytes
        df['POSIX_SHARED_BYTES_PERC'] = df.POSIX_SHARED_BYTES / total_bytes
        df['POSIX_READ_ONLY_BYTES_PERC'] = df.POSIX_READ_ONLY_BYTES / total_bytes
        df['POSIX_READ_WRITE_BYTES_PERC'] = df.POSIX_READ_WRITE_BYTES / total_bytes
        df['POSIX_WRITE_ONLY_BYTES_PERC'] = df.POSIX_WRITE_ONLY_BYTES / total_bytes
        df = df.drop(columns=["POSIX_BYTES_READ", "POSIX_UNIQUE_BYTES", "POSIX_SHARED_BYTES",
                              "POSIX_READ_ONLY_BYTES", "POSIX_READ_WRITE_BYTES", "POSIX_WRITE_ONLY_BYTES"])
    except:
        logging.error(
            "Failed to normalize one of the features in [POSIX_BYTES_READ, POSIX_BYTES_WRITTEN, unique_bytes, shared_bytes, read_only_bytes, read_write_bytes, write_only_bytes")

    try:
        df['POSIX_UNIQUE_FILES_PERC'] = df.POSIX_UNIQUE_FILES / total_files
        df['POSIX_SHARED_FILES_PERC'] = df.POSIX_SHARED_FILES / total_files
        df['POSIX_READ_ONLY_FILES_PERC'] = df.POSIX_READ_ONLY_FILES / total_files
        df['POSIX_READ_WRITE_FILES_PERC'] = df.POSIX_READ_WRITE_FILES / total_files
        df['POSIX_WRITE_ONLY_FILES_PERC'] = df.POSIX_WRITE_ONLY_FILES / total_files
        df = df.drop(
            columns=['POSIX_UNIQUE_FILES', 'POSIX_SHARED_FILES', 'POSIX_READ_ONLY_FILES', 'POSIX_READ_WRITE_FILES',
                     'POSIX_WRITE_ONLY_FILES'])
    except:
        logging.error("Failed to normalize one of the *_files features")

    try:
        df['POSIX_READS_PERC'] = df.POSIX_READS / total_accesses
        df['POSIX_WRITES_PERC'] = df.POSIX_WRITES / total_accesses
        df['POSIX_RW_SWITCHES_PERC'] = df.POSIX_RW_SWITCHES / total_accesses
        df['POSIX_SEQ_READS_PERC'] = df.POSIX_SEQ_READS / total_accesses
        df['POSIX_SEQ_WRITES_PERC'] = df.POSIX_SEQ_WRITES / total_accesses
        df['POSIX_CONSEC_READS_PERC'] = df.POSIX_CONSEC_READS / total_accesses
        df['POSIX_CONSEC_WRITES_PERC'] = df.POSIX_CONSEC_WRITES / total_accesses
        df['POSIX_FILE_NOT_ALIGNED_PERC'] = df.POSIX_FILE_NOT_ALIGNED / total_accesses
        df['POSIX_MEM_NOT_ALIGNED_PERC'] = df.POSIX_MEM_NOT_ALIGNED / total_accesses
        df = df.drop(columns=["POSIX_READS", "POSIX_WRITES", "POSIX_RW_SWITCHES", "POSIX_SEQ_WRITES", "POSIX_SEQ_READS",
                              "POSIX_CONSEC_READS", "POSIX_CONSEC_WRITES", "POSIX_FILE_NOT_ALIGNED",
                              "POSIX_MEM_NOT_ALIGNED"])
    except:
        logging.error(
            "Failed to normalize one of the features in [POSIX_READS, POSIX_WRITES, POSIX_SEQ_WRITES, POSIX_SEQ_READS, POSIX_CONSEC_READS, POSIX_CONSEC_WRITES, POSIX_FILE_NOT_ALIGNED_PERC, POSIX_MEM_NOT_ALIGNED_PERC]")

    try:
        if np.any(
                df.POSIX_SIZE_READ_0_100 + df.POSIX_SIZE_READ_100_1K + df.POSIX_SIZE_READ_1K_10K + df.POSIX_SIZE_READ_10K_100K +
                df.POSIX_SIZE_READ_100K_1M + df.POSIX_SIZE_READ_1M_4M + df.POSIX_SIZE_READ_4M_10M + df.POSIX_SIZE_READ_10M_100M +
                df.POSIX_SIZE_READ_100M_1G + df.POSIX_SIZE_READ_1G_PLUS +
                df.POSIX_SIZE_WRITE_0_100 + df.POSIX_SIZE_WRITE_100_1K + df.POSIX_SIZE_WRITE_1K_10K + df.POSIX_SIZE_WRITE_10K_100K +
                df.POSIX_SIZE_WRITE_100K_1M + df.POSIX_SIZE_WRITE_1M_4M + df.POSIX_SIZE_WRITE_4M_10M + df.POSIX_SIZE_WRITE_10M_100M +
                df.POSIX_SIZE_WRITE_100M_1G + df.POSIX_SIZE_WRITE_1G_PLUS != total_accesses):
            logging.warning("POSIX_SIZE_WRITE* + POSIX_SIZE_READ* columns do not add up to POSIX_total_accesses")

        df['POSIX_SIZE_READ_0_100_PERC'] = df.POSIX_SIZE_READ_0_100 / total_accesses
        df['POSIX_SIZE_READ_100_1K_PERC'] = df.POSIX_SIZE_READ_100_1K / total_accesses
        df['POSIX_SIZE_READ_1K_10K_PERC'] = df.POSIX_SIZE_READ_1K_10K / total_accesses
        df['POSIX_SIZE_READ_10K_100K_PERC'] = df.POSIX_SIZE_READ_10K_100K / total_accesses
        df['POSIX_SIZE_READ_100K_1M_PERC'] = df.POSIX_SIZE_READ_100K_1M / total_accesses
        df['POSIX_SIZE_READ_1M_4M_PERC'] = df.POSIX_SIZE_READ_1M_4M / total_accesses
        df['POSIX_SIZE_READ_4M_10M_PERC'] = df.POSIX_SIZE_READ_4M_10M / total_accesses
        df['POSIX_SIZE_READ_10M_100M_PERC'] = df.POSIX_SIZE_READ_10M_100M / total_accesses
        df['POSIX_SIZE_READ_100M_1G_PERC'] = df.POSIX_SIZE_READ_100M_1G / total_accesses
        df['POSIX_SIZE_READ_1G_PLUS_PERC'] = df.POSIX_SIZE_READ_1G_PLUS / total_accesses

        df['POSIX_SIZE_WRITE_0_100_PERC'] = df.POSIX_SIZE_WRITE_0_100 / total_accesses
        df['POSIX_SIZE_WRITE_100_1K_PERC'] = df.POSIX_SIZE_WRITE_100_1K / total_accesses
        df['POSIX_SIZE_WRITE_1K_10K_PERC'] = df.POSIX_SIZE_WRITE_1K_10K / total_accesses
        df['POSIX_SIZE_WRITE_10K_100K_PERC'] = df.POSIX_SIZE_WRITE_10K_100K / total_accesses
        df['POSIX_SIZE_WRITE_100K_1M_PERC'] = df.POSIX_SIZE_WRITE_100K_1M / total_accesses
        df['POSIX_SIZE_WRITE_1M_4M_PERC'] = df.POSIX_SIZE_WRITE_1M_4M / total_accesses
        df['POSIX_SIZE_WRITE_4M_10M_PERC'] = df.POSIX_SIZE_WRITE_4M_10M / total_accesses
        df['POSIX_SIZE_WRITE_10M_100M_PERC'] = df.POSIX_SIZE_WRITE_10M_100M / total_accesses
        df['POSIX_SIZE_WRITE_100M_1G_PERC'] = df.POSIX_SIZE_WRITE_100M_1G / total_accesses
        df['POSIX_SIZE_WRITE_1G_PLUS_PERC'] = df.POSIX_SIZE_WRITE_1G_PLUS / total_accesses

        drop_columns = ["POSIX_SIZE_READ_0_100", "POSIX_SIZE_READ_100_1K", "POSIX_SIZE_READ_1K_10K",
                        "POSIX_SIZE_READ_10K_100K",
                        "POSIX_SIZE_READ_100K_1M", "POSIX_SIZE_READ_1M_4M", "POSIX_SIZE_READ_4M_10M",
                        "POSIX_SIZE_READ_10M_100M",
                        "POSIX_SIZE_READ_100M_1G", "POSIX_SIZE_READ_1G_PLUS",
                        "POSIX_SIZE_WRITE_0_100", "POSIX_SIZE_WRITE_100_1K", "POSIX_SIZE_WRITE_1K_10K",
                        "POSIX_SIZE_WRITE_10K_100K",
                        "POSIX_SIZE_WRITE_100K_1M", "POSIX_SIZE_WRITE_1M_4M", "POSIX_SIZE_WRITE_4M_10M",
                        "POSIX_SIZE_WRITE_10M_100M",
                        "POSIX_SIZE_WRITE_100M_1G", "POSIX_SIZE_WRITE_1G_PLUS"]

        df = df.drop(columns=drop_columns)
    except:
        logging.warning("Failed to normalize POSIX_SIZE_*")

    try:
        df['POSIX_ACCESS1_COUNT_PERC'] = df.POSIX_ACCESS1_COUNT / total_accesses
        df['POSIX_ACCESS2_COUNT_PERC'] = df.POSIX_ACCESS2_COUNT / total_accesses
        df['POSIX_ACCESS3_COUNT_PERC'] = df.POSIX_ACCESS3_COUNT / total_accesses
        df['POSIX_ACCESS4_COUNT_PERC'] = df.POSIX_ACCESS4_COUNT / total_accesses

        logging.info("Normalized access values:")
        logging.info("Access 1 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS1_COUNT_PERC),
                                                                     np.mean(df.POSIX_ACCESS1_COUNT_PERC),
                                                                     np.median(df.POSIX_ACCESS1_COUNT_PERC)))
        logging.info("Access 2 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS2_COUNT_PERC),
                                                                     np.mean(df.POSIX_ACCESS2_COUNT_PERC),
                                                                     np.median(df.POSIX_ACCESS2_COUNT_PERC)))
        logging.info("Access 3 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS3_COUNT_PERC),
                                                                     np.mean(df.POSIX_ACCESS3_COUNT_PERC),
                                                                     np.median(df.POSIX_ACCESS3_COUNT_PERC)))
        logging.info("Access 4 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS4_COUNT_PERC),
                                                                     np.mean(df.POSIX_ACCESS4_COUNT_PERC),
                                                                     np.median(df.POSIX_ACCESS4_COUNT_PERC)))

        df = df.drop(
            columns=['POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT', 'POSIX_ACCESS3_COUNT', 'POSIX_ACCESS4_COUNT'])
    except:
        logging.warning("Failed to normalize POSIX_ACCESS[1-4]_COUNT")

    # In case of division by zero, we'll get NaN. We convert those to zeros.
    df = df.fillna(0)

    if remove_dual:
        df = df.drop(columns=['POSIX_BYTES_WRITTEN_PERC', 'POSIX_SHARED_BYTES_PERC', 'POSIX_READ_WRITE_BYTES_PERC',
                              'POSIX_READ_WRITE_FILES_PERC', 'POSIX_WRITES_PERC', 'POSIX_SHARED_FILES_PERC'])

    # 重新添加NODES列
    if nodes_data is not None:
        df['NODES'] = nodes_data

    # 确保NODES列在正确的位置
    if 'NODES' in df.columns:
        cols = ['NODES'] + [col for col in df.columns if col != 'NODES']
        df = df[cols]

    return df


def log_scale_dataset(df, add_small_value=1.1, set_NaNs_to=-10):
    """
    Takes the log10 of a DF + a small value (to prevent -infs),
    and replaces NaN values with a predetermined value.
    Adds the new columns to the dataset, and renames the original ones.
    """
    number_columns = get_number_columns(df)
    columns_ = [x for x in number_columns if "perc" not in x.lower()]
    logging.info("Applying log10() to the columns {}".format(columns_))

    # 保存NODES列
    nodes_data = None
    if 'NODES' in df.columns:
        nodes_data = df['NODES'].copy()
        df = df.drop(columns=['NODES'])

    # 对非NODES列进行log10转换
    for c in columns_:
        if c != 'NODES':  # 跳过NODES列
            df[c + "_LOG10"] = np.log10(df[c] + add_small_value).fillna(value=set_NaNs_to)

    # 删除原始列
    columns_to_drop = [x for x in columns_ if x != 'NODES']
    df = df.drop(columns=columns_to_drop)

    # 重新添加NODES列
    if nodes_data is not None:
        df['NODES'] = nodes_data

    # 确保NODES列在正确的位置
    if 'NODES' in df.columns:
        cols = ['NODES'] + [col for col in df.columns if col != 'NODES']
        df = df[cols]

    return df


def extracting_log(path):
    # 使用从feature.py导入的log_features
    POSIX_OPENS = []
    POSIX_SEEKS = []
    POSIX_STATS = []
    POSIX_MMAPS = []
    POSIX_FSYNCS = []
    POSIX_MODE = []
    POSIX_MEM_ALIGNMENT = []
    POSIX_FILE_ALIGNMENT = []
    NPROCS = []
    NODES = []
    POSIX_BYTES_READ = []
    POSIX_BYTES_WRITTEN = []
    POSIX_UNIQUE_FILES = []
    POSIX_SHARED_FILES = []
    POSIX_READS = []
    POSIX_WRITES = []

    POSIX_OPENS_PATTERN = '(total_POSIX_OPENS:\s+)(\d+)'
    ds_opens_pattern = re.compile(POSIX_OPENS_PATTERN)

    POSIX_SEEKS_PATTERN = '(total_POSIX_SEEKS:\s+)(\d+)'
    ds_seeks_pattern = re.compile(POSIX_SEEKS_PATTERN)

    POSIX_STATS_PATTERN = '(total_POSIX_STATS:\s+)(\d+)'
    ds_stats_pattern = re.compile(POSIX_STATS_PATTERN)

    POSIX_MMAPS_PATTERN = '(total_POSIX_MMAPS:\s+)(\S+)'
    ds_mmaps_pattern = re.compile(POSIX_MMAPS_PATTERN)

    POSIX_FSYNCS_PATTERN = '(total_POSIX_FSYNCS:\s+)(\d+)'
    ds_fsyncs_pattern = re.compile(POSIX_FSYNCS_PATTERN)

    POSIX_MODE_PATTERN = '(total_POSIX_MODE:\s+)(\d+)'
    ds_mode_pattern = re.compile(POSIX_MODE_PATTERN)

    POSIX_MEM_ALIGN_PATTERN = '(total_POSIX_MEM_ALIGNMENT:\s+)(\d+)'
    ds_mem_align_pattern = re.compile(POSIX_MEM_ALIGN_PATTERN)

    POSIX_FILE_ALIGN_PATTERN = '(total_POSIX_FILE_ALIGNMENT:\s+)(\d+)'
    ds_file_align_pattern = re.compile(POSIX_FILE_ALIGN_PATTERN)

    DARSHAN_NPROCS_PATTERN = '(#\s+nprocs:\s+)(\d+)'
    ds_nprocs_pattern = re.compile(DARSHAN_NPROCS_PATTERN)

    POSIX_READS_PATTERN = '(total_POSIX_READS:\s+)(\d+)'
    ds_reads_pattern = re.compile(POSIX_READS_PATTERN)

    POSIX_WRITES_PATTERN = '(total_POSIX_WRITES:\s+)(\d+)'
    ds_writes_pattern = re.compile(POSIX_WRITES_PATTERN)

    POSIX_BYTES_READ_PATTERN = '(total_POSIX_BYTES_READ:\s+)(\d+)'
    ds_bytes_read_pattern = re.compile(POSIX_BYTES_READ_PATTERN)

    POSIX_BYTES_WRITTEN_PATTERN = '(total_POSIX_BYTES_WRITTEN:\s+)(\d+)'
    ds_bytes_written_pattern = re.compile(POSIX_BYTES_WRITTEN_PATTERN)

    POSIX_UNIQUE_FILES_PATTERN = '(#\s+unique:\s+)(\d+)(\s)(\d+)(\s)(\d+)'
    ds_unique_files_pattern = re.compile(POSIX_UNIQUE_FILES_PATTERN)

    POSIX_SHARED_FILES_PATTERN = '(#\s+shared:\s+)(\d+)(\s)(\d+)(\s)(\d+)'
    ds_shared_files_pattern = re.compile(POSIX_SHARED_FILES_PATTERN)

    NODES_PATTERN = r'N-(\d+)_'
    ds_nodes_pattern = re.compile(NODES_PATTERN)

    darshan_files = glob.glob(path)

    for file_name in darshan_files:
        # 提取节点规模
        nodes_match = ds_nodes_pattern.search(file_name)
        if nodes_match is not None:
            nodes = int(nodes_match.group(1))
            NODES.append(nodes)
        else:
            raise
            NODES.append(0)

        with open(file_name) as infile:
            for line in infile:
                if line == '# MPI-IO module data\n':
                    break
                opens_match = ds_opens_pattern.match(line)
                if opens_match is not None:
                    posix_opens = int(opens_match.group(2))
                    POSIX_OPENS.append(posix_opens)
                    continue

                seeks_match = ds_seeks_pattern.match(line)
                if seeks_match is not None:
                    posix_seeks = int(seeks_match.group(2))
                    POSIX_SEEKS.append(posix_seeks)
                    continue

                stats_match = ds_stats_pattern.match(line)
                if stats_match is not None:
                    posix_stats = int(stats_match.group(2))
                    POSIX_STATS.append(posix_stats)
                    continue

                mmaps_match = ds_mmaps_pattern.match(line)
                if mmaps_match is not None:
                    posix_mmaps = int(mmaps_match.group(2))
                    POSIX_MMAPS.append(posix_mmaps)
                    continue

                fsyncs_match = ds_fsyncs_pattern.match(line)
                if fsyncs_match is not None:
                    posix_fsyncs = int(fsyncs_match.group(2))
                    POSIX_FSYNCS.append(posix_fsyncs)
                    continue

                mode_match = ds_mode_pattern.match(line)
                if mode_match is not None:
                    posix_mode = int(mode_match.group(2))
                    POSIX_MODE.append(posix_mode)
                    continue

                mem_align_match = ds_mem_align_pattern.match(line)
                if mem_align_match is not None:
                    posix_mem_align = int(mem_align_match.group(2))
                    POSIX_MEM_ALIGNMENT.append(posix_mem_align)
                    continue

                file_align_match = ds_file_align_pattern.match(line)
                if file_align_match is not None:
                    posix_file_align = int(file_align_match.group(2))
                    POSIX_FILE_ALIGNMENT.append(posix_file_align)
                    continue

                nprocs_match = ds_nprocs_pattern.match(line)
                if nprocs_match is not None:
                    nprocs = int(nprocs_match.group(2))
                    NPROCS.append(nprocs)
                    continue

                reads_match = ds_reads_pattern.match(line)
                if reads_match is not None:
                    posix_reads = int(reads_match.group(2))
                    POSIX_READS.append(posix_reads)
                    continue

                writes_match = ds_writes_pattern.match(line)
                if writes_match is not None:
                    posix_writes = int(writes_match.group(2))
                    POSIX_WRITES.append(posix_writes)
                    continue

                bytes_read_match = ds_bytes_read_pattern.match(line)
                if bytes_read_match is not None:
                    posix_bytes_read = int(bytes_read_match.group(2))
                    POSIX_BYTES_READ.append(posix_bytes_read)
                    continue

                bytes_written_match = ds_bytes_written_pattern.match(line)
                if bytes_written_match is not None:
                    posix_bytes_written = int(bytes_written_match.group(2))
                    POSIX_BYTES_WRITTEN.append(posix_bytes_written)
                    continue

                unique_files_match = ds_unique_files_pattern.match(line)
                if unique_files_match is not None:
                    posix_unique_files = int(unique_files_match.group(2))
                    posix_unique_bytes = int(unique_files_match.group(4))
                    POSIX_UNIQUE_FILES.append(posix_unique_files)
                    continue

                shared_files_match = ds_shared_files_pattern.match(line)
                if shared_files_match is not None:
                    posix_shared_files = int(shared_files_match.group(2))
                    posix_shared_bytes = int(shared_files_match.group(4))
                    POSIX_SHARED_FILES.append(posix_shared_files)
                    continue

    POSIX_TOTAL_ACCESSES = column_sum(POSIX_READS, POSIX_WRITES)
    POSIX_TOTAL_BYTES = column_sum(POSIX_BYTES_READ, POSIX_BYTES_WRITTEN)
    POSIX_TOTAL_FILES = column_sum(POSIX_SHARED_FILES, POSIX_UNIQUE_FILES)

    mydata = pd.DataFrame(list(
        zip(POSIX_OPENS, POSIX_SEEKS, POSIX_STATS, POSIX_MMAPS, POSIX_FSYNCS, POSIX_MODE, POSIX_MEM_ALIGNMENT,
            POSIX_FILE_ALIGNMENT, NPROCS, POSIX_TOTAL_ACCESSES, POSIX_TOTAL_BYTES, POSIX_TOTAL_FILES, NODES)),
        columns=log_features)
    return mydata


def extracting_perc1(path):
    # 使用从feature.py导入的perc_features1
    POSIX_BYTES_READ = []
    POSIX_BYTES_WRITTEN = []
    POSIX_UNIQUE_BYTES = []
    POSIX_SHARED_BYTES = []
    POSIX_READ_ONLY_BYTES = []
    POSIX_READ_WRITE_BYTES = []
    POSIX_WRITE_ONLY_BYTES = []
    POSIX_UNIQUE_FILES = []
    POSIX_SHARED_FILES = []
    POSIX_READ_ONLY_FILES = []
    POSIX_WRITE_ONLY_FILES = []
    POSIX_READS = []
    POSIX_WRITES = []
    POSIX_READ_WRITE_FILES = []
    POSIX_RW_SWITCHES = []
    POSIX_SEQ_READS = []
    POSIX_SEQ_WRITES = []
    POSIX_CONSEC_READS = []
    POSIX_CONSEC_WRITES = []
    POSIX_FILE_NOT_ALIGNED = []
    POSIX_MEM_NOT_ALIGNED = []
    POSIX_ACCESS1_COUNT = []
    POSIX_ACCESS2_COUNT = []
    POSIX_ACCESS3_COUNT = []
    POSIX_ACCESS4_COUNT = []

    POSIX_READS_PATTERN = '(total_POSIX_READS:\s+)(\d+)'
    ds_reads_pattern = re.compile(POSIX_READS_PATTERN)

    POSIX_WRITES_PATTERN = '(total_POSIX_WRITES:\s+)(\d+)'
    ds_writes_pattern = re.compile(POSIX_WRITES_PATTERN)

    POSIX_BYTES_READ_PATTERN = '(total_POSIX_BYTES_READ:\s+)(\d+)'
    ds_bytes_read_pattern = re.compile(POSIX_BYTES_READ_PATTERN)

    POSIX_BYTES_WRITTEN_PATTERN = '(total_POSIX_BYTES_WRITTEN:\s+)(\d+)'
    ds_bytes_written_pattern = re.compile(POSIX_BYTES_WRITTEN_PATTERN)

    POSIX_UNIQUE_FILES_PATTERN = '(#\s+unique:\s+)(\d+)(\s)(\d+)(\s)(\d+)'
    ds_unique_files_pattern = re.compile(POSIX_UNIQUE_FILES_PATTERN)

    POSIX_SHARED_FILES_PATTERN = '(#\s+shared:\s+)(\d+)(\s)(\d+)(\s)(\d+)'
    ds_shared_files_pattern = re.compile(POSIX_SHARED_FILES_PATTERN)

    POSIX_READ_ONLY_FILES_PATTERN = '(#\s+read_only:\s+)(\d+)(\s)(\d+)(\s)(\d+)'
    ds_read_only_files_pattern = re.compile(POSIX_READ_ONLY_FILES_PATTERN)

    POSIX_WRITE_ONLY_FILES_PATTERN = '(#\s+write_only:\s+)(\d+)(\s)(\d+)(\s)(\d+)'
    ds_write_only_files_pattern = re.compile(POSIX_WRITE_ONLY_FILES_PATTERN)

    POSIX_READ_WRITE_FILES_PATTERN = '(#\s+read_write:\s+)(\d+)(\s)(\d+)(\s)(\d+)'
    ds_read_write_files_pattern = re.compile(POSIX_READ_WRITE_FILES_PATTERN)

    POSIX_RW_SWITCHES_PATTERN = '(total_POSIX_RW_SWITCHES:\s+)(\d+)'
    ds_rw_switches_pattern = re.compile(POSIX_RW_SWITCHES_PATTERN)

    POSIX_SEQ_READS_PATTERN = '(total_POSIX_SEQ_READS:\s+)(\d+)'
    ds_seq_reads_pattern = re.compile(POSIX_SEQ_READS_PATTERN)

    POSIX_SEQ_WRITES_PATTERN = '(total_POSIX_SEQ_WRITES:\s+)(\d+)'
    ds_seq_writes_pattern = re.compile(POSIX_SEQ_WRITES_PATTERN)

    POSIX_CONSEC_READS_PATTERN = '(total_POSIX_CONSEC_READS:\s+)(\d+)'
    ds_consec_reads_pattern = re.compile(POSIX_CONSEC_READS_PATTERN)

    POSIX_CONSEC_WRITES_PATTERN = '(total_POSIX_CONSEC_WRITES:\s+)(\d+)'
    ds_consec_writes_pattern = re.compile(POSIX_CONSEC_WRITES_PATTERN)

    POSIX_MEM_NOT_ALIGN_PATTERN = '(total_POSIX_MEM_NOT_ALIGNED:\s+)(\d+)'
    ds_mem_not_align_pattern = re.compile(POSIX_MEM_NOT_ALIGN_PATTERN)

    POSIX_FILE_NOT_ALIGN_PATTERN = '(total_POSIX_FILE_NOT_ALIGNED:\s+)(\d+)'
    ds_file_not_align_pattern = re.compile(POSIX_FILE_NOT_ALIGN_PATTERN)

    POSIX_ACCESS1_PATTERN = '(total_POSIX_ACCESS1_COUNT:\s+)(\d+)'
    ds_access1_pattern = re.compile(POSIX_ACCESS1_PATTERN)

    POSIX_ACCESS2_PATTERN = '(total_POSIX_ACCESS2_COUNT:\s+)(\d+)'
    ds_access2_pattern = re.compile(POSIX_ACCESS2_PATTERN)

    POSIX_ACCESS3_PATTERN = '(total_POSIX_ACCESS3_COUNT:\s+)(\d+)'
    ds_access3_pattern = re.compile(POSIX_ACCESS3_PATTERN)

    POSIX_ACCESS4_PATTERN = '(total_POSIX_ACCESS4_COUNT:\s+)(\d+)'
    ds_access4_pattern = re.compile(POSIX_ACCESS4_PATTERN)

    darshan_files = glob.glob(path)
    for file_name in darshan_files:
        with open(file_name) as infile:
            for line in infile:
                if line == '# MPI-IO module data\n':
                    break
                reads_match = ds_reads_pattern.match(line)
                if reads_match is not None:
                    posix_reads = int(reads_match.group(2))
                    POSIX_READS.append(posix_reads)
                    continue

                writes_match = ds_writes_pattern.match(line)
                if writes_match is not None:
                    posix_writes = int(writes_match.group(2))
                    POSIX_WRITES.append(posix_writes)
                    continue

                bytes_read_match = ds_bytes_read_pattern.match(line)
                if bytes_read_match is not None:
                    posix_bytes_read = int(bytes_read_match.group(2))
                    POSIX_BYTES_READ.append(posix_bytes_read)
                    continue

                bytes_written_match = ds_bytes_written_pattern.match(line)
                if bytes_written_match is not None:
                    posix_bytes_written = int(bytes_written_match.group(2))
                    POSIX_BYTES_WRITTEN.append(posix_bytes_written)
                    continue

                unique_files_match = ds_unique_files_pattern.match(line)
                if unique_files_match is not None:
                    posix_unique_files = int(unique_files_match.group(2))
                    posix_unique_bytes = int(unique_files_match.group(4))
                    POSIX_UNIQUE_FILES.append(posix_unique_files)
                    POSIX_UNIQUE_BYTES.append(posix_unique_bytes)
                    continue

                shared_files_match = ds_shared_files_pattern.match(line)
                if shared_files_match is not None:
                    posix_shared_files = int(shared_files_match.group(2))
                    posix_shared_bytes = int(shared_files_match.group(4))
                    POSIX_SHARED_FILES.append(posix_shared_files)
                    POSIX_SHARED_BYTES.append(posix_shared_bytes)
                    continue

                read_only_files_match = ds_read_only_files_pattern.match(line)
                if read_only_files_match is not None:
                    posix_read_only_files = int(read_only_files_match.group(2))
                    posix_read_only_bytes = int(read_only_files_match.group(4))
                    POSIX_READ_ONLY_FILES.append(posix_read_only_files)
                    POSIX_READ_ONLY_BYTES.append(posix_read_only_bytes)
                    continue

                write_only_files_match = ds_write_only_files_pattern.match(line)
                if write_only_files_match is not None:
                    posix_write_only_files = int(write_only_files_match.group(2))
                    posix_write_only_bytes = int(write_only_files_match.group(4))
                    POSIX_WRITE_ONLY_FILES.append(posix_write_only_files)
                    POSIX_WRITE_ONLY_BYTES.append(posix_write_only_bytes)
                    continue

                read_write_files_match = ds_read_write_files_pattern.match(line)
                if read_write_files_match is not None:
                    posix_read_write_files = int(read_write_files_match.group(2))
                    posix_read_write_bytes = int(read_write_files_match.group(4))
                    POSIX_READ_WRITE_FILES.append(posix_read_write_files)
                    POSIX_READ_WRITE_BYTES.append(posix_read_write_bytes)
                    continue

                rw_switches_match = ds_rw_switches_pattern.match(line)
                if rw_switches_match is not None:
                    posix_rw_switches = int(rw_switches_match.group(2))
                    POSIX_RW_SWITCHES.append(posix_rw_switches)
                    continue

                seq_reads_match = ds_seq_reads_pattern.match(line)
                if seq_reads_match is not None:
                    posix_seq_reads = int(seq_reads_match.group(2))
                    POSIX_SEQ_READS.append(posix_seq_reads)
                    continue

                seq_writes_match = ds_seq_writes_pattern.match(line)
                if seq_writes_match is not None:
                    posix_seq_writes = int(seq_writes_match.group(2))
                    POSIX_SEQ_WRITES.append(posix_seq_writes)
                    continue

                consec_reads_match = ds_consec_reads_pattern.match(line)
                if consec_reads_match is not None:
                    posix_consec_reads = int(consec_reads_match.group(2))
                    POSIX_CONSEC_READS.append(posix_consec_reads)
                    continue

                consec_writes_match = ds_consec_writes_pattern.match(line)
                if consec_writes_match is not None:
                    posix_consec_writes = int(consec_writes_match.group(2))
                    POSIX_CONSEC_WRITES.append(posix_consec_writes)
                    continue

                mem_not_align_match = ds_mem_not_align_pattern.match(line)
                if mem_not_align_match is not None:
                    posix_mem_not_aligned = int(mem_not_align_match.group(2))
                    POSIX_MEM_NOT_ALIGNED.append(posix_mem_not_aligned)
                    continue

                file_not_align_match = ds_file_not_align_pattern.match(line)
                if file_not_align_match is not None:
                    posix_file_not_aligned = int(file_not_align_match.group(2))
                    POSIX_FILE_NOT_ALIGNED.append(posix_file_not_aligned)
                    continue

                access1_match = ds_access1_pattern.match(line)
                if access1_match is not None:
                    posix_access1 = int(access1_match.group(2))
                    POSIX_ACCESS1_COUNT.append(posix_access1)
                    continue

                access2_match = ds_access2_pattern.match(line)
                if access2_match is not None:
                    posix_access2 = int(access2_match.group(2))
                    POSIX_ACCESS2_COUNT.append(posix_access2)
                    continue

                access3_match = ds_access3_pattern.match(line)
                if access3_match is not None:
                    posix_access3 = int(access3_match.group(2))
                    POSIX_ACCESS3_COUNT.append(posix_access3)
                    continue

                access4_match = ds_access4_pattern.match(line)
                if access4_match is not None:
                    posix_access4 = int(access4_match.group(2))
                    POSIX_ACCESS4_COUNT.append(posix_access4)
                    continue

    mydata = pd.DataFrame(list(
        zip(POSIX_BYTES_READ, POSIX_UNIQUE_BYTES, POSIX_SHARED_BYTES, POSIX_READ_ONLY_BYTES, POSIX_READ_WRITE_BYTES,
            POSIX_WRITE_ONLY_BYTES, POSIX_UNIQUE_FILES, POSIX_SHARED_FILES, POSIX_READ_ONLY_FILES,
            POSIX_READ_WRITE_FILES, POSIX_WRITE_ONLY_FILES, POSIX_READS, POSIX_WRITES, POSIX_RW_SWITCHES,
            POSIX_SEQ_READS, POSIX_SEQ_WRITES, POSIX_CONSEC_READS, POSIX_CONSEC_WRITES, POSIX_FILE_NOT_ALIGNED,
            POSIX_MEM_NOT_ALIGNED, POSIX_ACCESS1_COUNT, POSIX_ACCESS2_COUNT, POSIX_ACCESS3_COUNT, POSIX_ACCESS4_COUNT)),
        columns=perc_features1)
    # print(mydata)
    return mydata


def extracting_perc2(path):
    # 使用从feature.py导入的perc_features2
    POSIX_SIZE_READ_0_100 = []
    POSIX_SIZE_READ_100_1K = []
    POSIX_SIZE_READ_1K_10K = []
    POSIX_SIZE_READ_10K_100K = []
    POSIX_SIZE_READ_100K_1M = []
    POSIX_SIZE_READ_1M_4M = []
    POSIX_SIZE_READ_4M_10M = []
    POSIX_SIZE_READ_10M_100M = []
    POSIX_SIZE_READ_100M_1G = []
    POSIX_SIZE_READ_1G_PLUS = []
    POSIX_SIZE_WRITE_0_100 = []
    POSIX_SIZE_WRITE_100_1K = []
    POSIX_SIZE_WRITE_1K_10K = []
    POSIX_SIZE_WRITE_10K_100K = []
    POSIX_SIZE_WRITE_100K_1M = []
    POSIX_SIZE_WRITE_1M_4M = []
    POSIX_SIZE_WRITE_4M_10M = []
    POSIX_SIZE_WRITE_10M_100M = []
    POSIX_SIZE_WRITE_100M_1G = []
    POSIX_SIZE_WRITE_1G_PLUS = []

    perc_features2 = ['POSIX_SIZE_READ_0_100',
                      'POSIX_SIZE_READ_100_1K', 'POSIX_SIZE_READ_1K_10K',
                      'POSIX_SIZE_READ_10K_100K', 'POSIX_SIZE_READ_100K_1M',
                      'POSIX_SIZE_READ_1M_4M', 'POSIX_SIZE_READ_4M_10M',
                      'POSIX_SIZE_READ_10M_100M', 'POSIX_SIZE_READ_100M_1G',
                      'POSIX_SIZE_READ_1G_PLUS', 'POSIX_SIZE_WRITE_0_100',
                      'POSIX_SIZE_WRITE_100_1K', 'POSIX_SIZE_WRITE_1K_10K',
                      'POSIX_SIZE_WRITE_10K_100K', 'POSIX_SIZE_WRITE_100K_1M',
                      'POSIX_SIZE_WRITE_1M_4M', 'POSIX_SIZE_WRITE_4M_10M',
                      'POSIX_SIZE_WRITE_10M_100M', 'POSIX_SIZE_WRITE_100M_1G',
                      'POSIX_SIZE_WRITE_1G_PLUS', ]

    POSIX_SIZE_READ_0_100_PATTERN = '(total_POSIX_SIZE_READ_0_100:\s+)(\d+)'
    ds_read_0_100_pattern = re.compile(POSIX_SIZE_READ_0_100_PATTERN)

    POSIX_SIZE_READ_100_1K_PATTERN = '(total_POSIX_SIZE_READ_100_1K:\s+)(\d+)'
    ds_read_100_1k_pattern = re.compile(POSIX_SIZE_READ_100_1K_PATTERN)

    POSIX_SIZE_READ_1K_10K_PATTERN = '(total_POSIX_SIZE_READ_1K_10K:\s+)(\d+)'
    ds_read_1k_10k_pattern = re.compile(POSIX_SIZE_READ_1K_10K_PATTERN)

    POSIX_SIZE_READ_10K_100K_PATTERN = '(total_POSIX_SIZE_READ_10K_100K:\s+)(\d+)'
    ds_read_10k_100k_pattern = re.compile(POSIX_SIZE_READ_10K_100K_PATTERN)

    POSIX_SIZE_READ_100K_1M_PATTERN = '(total_POSIX_SIZE_READ_100K_1M:\s+)(\d+)'
    ds_read_100k_1m_pattern = re.compile(POSIX_SIZE_READ_100K_1M_PATTERN)

    POSIX_SIZE_READ_1M_4M_PATTERN = '(total_POSIX_SIZE_READ_1M_4M:\s+)(\d+)'
    ds_read_1m_4m_pattern = re.compile(POSIX_SIZE_READ_1M_4M_PATTERN)

    POSIX_SIZE_READ_4M_10M_PATTERN = '(total_POSIX_SIZE_READ_4M_10M:\s+)(\d+)'
    ds_read_4m_10m_pattern = re.compile(POSIX_SIZE_READ_4M_10M_PATTERN)

    POSIX_SIZE_READ_10M_100M_PATTERN = '(total_POSIX_SIZE_READ_10M_100M:\s+)(\d+)'
    ds_read_10m_100m_pattern = re.compile(POSIX_SIZE_READ_10M_100M_PATTERN)

    POSIX_SIZE_READ_100M_1G_PATTERN = '(total_POSIX_SIZE_READ_100M_1G:\s+)(\d+)'
    ds_read_100m_1g_pattern = re.compile(POSIX_SIZE_READ_100M_1G_PATTERN)

    POSIX_SIZE_READ_1G_PLUS_PATTERN = '(total_POSIX_SIZE_READ_1G_PLUS:\s+)(\d+)'
    ds_read_1g_plus_pattern = re.compile(POSIX_SIZE_READ_1G_PLUS_PATTERN)

    POSIX_SIZE_WRITE_0_100_PATTERN = '(total_POSIX_SIZE_WRITE_0_100:\s+)(\d+)'
    ds_write_0_100_pattern = re.compile(POSIX_SIZE_WRITE_0_100_PATTERN)

    POSIX_SIZE_WRITE_100_1K_PATTERN = '(total_POSIX_SIZE_WRITE_100_1K:\s+)(\d+)'
    ds_write_100_1k_pattern = re.compile(POSIX_SIZE_WRITE_100_1K_PATTERN)

    POSIX_SIZE_WRITE_1K_10K_PATTERN = '(total_POSIX_SIZE_WRITE_1K_10K:\s+)(\d+)'
    ds_write_1k_10k_pattern = re.compile(POSIX_SIZE_WRITE_1K_10K_PATTERN)

    POSIX_SIZE_WRITE_10K_100K_PATTERN = '(total_POSIX_SIZE_WRITE_10K_100K:\s+)(\d+)'
    ds_write_10k_100k_pattern = re.compile(POSIX_SIZE_WRITE_10K_100K_PATTERN)

    POSIX_SIZE_WRITE_100K_1M_PATTERN = '(total_POSIX_SIZE_WRITE_100K_1M:\s+)(\d+)'
    ds_write_100k_1m_pattern = re.compile(POSIX_SIZE_WRITE_100K_1M_PATTERN)

    POSIX_SIZE_WRITE_1M_4M_PATTERN = '(total_POSIX_SIZE_WRITE_1M_4M:\s+)(\d+)'
    ds_write_1m_4m_pattern = re.compile(POSIX_SIZE_WRITE_1M_4M_PATTERN)

    POSIX_SIZE_WRITE_4M_10M_PATTERN = '(total_POSIX_SIZE_WRITE_4M_10M:\s+)(\d+)'
    ds_write_4m_10m_pattern = re.compile(POSIX_SIZE_WRITE_4M_10M_PATTERN)

    POSIX_SIZE_WRITE_10M_100M_PATTERN = '(total_POSIX_SIZE_WRITE_10M_100M:\s+)(\d+)'
    ds_write_10m_100m_pattern = re.compile(POSIX_SIZE_WRITE_10M_100M_PATTERN)

    POSIX_SIZE_WRITE_100M_1G_PATTERN = '(total_POSIX_SIZE_WRITE_100M_1G:\s+)(\d+)'
    ds_write_100m_1g_pattern = re.compile(POSIX_SIZE_WRITE_100M_1G_PATTERN)

    POSIX_SIZE_WRITE_1G_PLUS_PATTERN = '(total_POSIX_SIZE_WRITE_1G_PLUS:\s+)(\d+)'
    ds_write_1g_plus_pattern = re.compile(POSIX_SIZE_WRITE_1G_PLUS_PATTERN)

    darshan_files = glob.glob(path)

    for file_name in darshan_files:
        with open(file_name) as infile:
            for line in infile:
                if line == '# MPI-IO module data\n':
                    break
                read_0_100_match = ds_read_0_100_pattern.match(line)
                if read_0_100_match is not None:
                    posix_read_0_100 = int(read_0_100_match.group(2))
                    POSIX_SIZE_READ_0_100.append(posix_read_0_100)
                    continue

                read_100_1k_match = ds_read_100_1k_pattern.match(line)
                if read_100_1k_match is not None:
                    posix_read_100_1k = int(read_100_1k_match.group(2))
                    POSIX_SIZE_READ_100_1K.append(posix_read_100_1k)
                    continue

                read_1k_10k_match = ds_read_1k_10k_pattern.match(line)
                if read_1k_10k_match is not None:
                    posix_read_1k_10k = int(read_1k_10k_match.group(2))
                    POSIX_SIZE_READ_1K_10K.append(posix_read_1k_10k)
                    continue

                read_10k_100k_match = ds_read_10k_100k_pattern.match(line)
                if read_10k_100k_match is not None:
                    posix_read_10k_100k = int(read_10k_100k_match.group(2))
                    POSIX_SIZE_READ_10K_100K.append(posix_read_10k_100k)
                    continue

                read_100k_1m_match = ds_read_100k_1m_pattern.match(line)
                if read_100k_1m_match is not None:
                    posix_read_100k_1m = int(read_100k_1m_match.group(2))
                    POSIX_SIZE_READ_100K_1M.append(posix_read_100k_1m)
                    continue

                read_1m_4m_match = ds_read_1m_4m_pattern.match(line)
                if read_1m_4m_match is not None:
                    posix_read_1m_4m = int(read_1m_4m_match.group(2))
                    POSIX_SIZE_READ_1M_4M.append(posix_read_1m_4m)
                    continue

                read_4m_10m_match = ds_read_4m_10m_pattern.match(line)
                if read_4m_10m_match is not None:
                    posix_read_4m_10m = int(read_4m_10m_match.group(2))
                    POSIX_SIZE_READ_4M_10M.append(posix_read_4m_10m)
                    continue

                read_10m_100m_match = ds_read_10m_100m_pattern.match(line)
                if read_10m_100m_match is not None:
                    posix_read_10m_100m = int(read_10m_100m_match.group(2))
                    POSIX_SIZE_READ_10M_100M.append(posix_read_10m_100m)
                    continue

                read_100m_1g_match = ds_read_100m_1g_pattern.match(line)
                if read_100m_1g_match is not None:
                    posix_read_100m_1g = int(read_100m_1g_match.group(2))
                    POSIX_SIZE_READ_100M_1G.append(posix_read_100m_1g)
                    continue

                read_1g_plus_match = ds_read_1g_plus_pattern.match(line)
                if read_1g_plus_match is not None:
                    posix_read_1g_plus = int(read_1g_plus_match.group(2))
                    POSIX_SIZE_READ_1G_PLUS.append(posix_read_1g_plus)
                    continue

                write_0_100_match = ds_write_0_100_pattern.match(line)
                if write_0_100_match is not None:
                    posix_write_0_100 = int(write_0_100_match.group(2))
                    POSIX_SIZE_WRITE_0_100.append(posix_write_0_100)
                    continue

                write_100_1k_match = ds_write_100_1k_pattern.match(line)
                if write_100_1k_match is not None:
                    posix_write_100_1k = int(write_100_1k_match.group(2))
                    POSIX_SIZE_WRITE_100_1K.append(posix_write_100_1k)
                    continue

                write_1k_10k_match = ds_write_1k_10k_pattern.match(line)
                if write_1k_10k_match is not None:
                    posix_write_1k_10k = int(write_1k_10k_match.group(2))
                    POSIX_SIZE_WRITE_1K_10K.append(posix_write_1k_10k)
                    continue

                write_10k_100k_match = ds_write_10k_100k_pattern.match(line)
                if write_10k_100k_match is not None:
                    posix_write_10k_100k = int(write_10k_100k_match.group(2))
                    POSIX_SIZE_WRITE_10K_100K.append(posix_write_10k_100k)
                    continue

                write_100k_1m_match = ds_write_100k_1m_pattern.match(line)
                if write_100k_1m_match is not None:
                    posix_write_100k_1m = int(write_100k_1m_match.group(2))
                    POSIX_SIZE_WRITE_100K_1M.append(posix_write_100k_1m)
                    continue

                write_1m_4m_match = ds_write_1m_4m_pattern.match(line)
                if write_1m_4m_match is not None:
                    posix_write_1m_4m = int(write_1m_4m_match.group(2))
                    POSIX_SIZE_WRITE_1M_4M.append(posix_write_1m_4m)
                    continue

                write_4m_10m_match = ds_write_4m_10m_pattern.match(line)
                if write_4m_10m_match is not None:
                    posix_write_4m_10m = int(write_4m_10m_match.group(2))
                    POSIX_SIZE_WRITE_4M_10M.append(posix_write_4m_10m)
                    continue

                write_10m_100m_match = ds_write_10m_100m_pattern.match(line)
                if write_10m_100m_match is not None:
                    posix_write_10m_100m = int(write_10m_100m_match.group(2))
                    POSIX_SIZE_WRITE_10M_100M.append(posix_write_10m_100m)
                    continue

                write_100m_1g_match = ds_write_100m_1g_pattern.match(line)
                if write_100m_1g_match is not None:
                    posix_write_100m_1g = int(write_100m_1g_match.group(2))
                    POSIX_SIZE_WRITE_100M_1G.append(posix_write_100m_1g)
                    continue

                write_1g_plus_match = ds_write_1g_plus_pattern.match(line)
                if write_1g_plus_match is not None:
                    posix_write_1g_plus = int(write_1g_plus_match.group(2))
                    POSIX_SIZE_WRITE_1G_PLUS.append(posix_write_1g_plus)
                    continue

    mydata = pd.DataFrame(list(
        zip(POSIX_SIZE_READ_0_100, POSIX_SIZE_READ_100_1K, POSIX_SIZE_READ_1K_10K, POSIX_SIZE_READ_10K_100K,
            POSIX_SIZE_READ_100K_1M, POSIX_SIZE_READ_1M_4M, POSIX_SIZE_READ_4M_10M, POSIX_SIZE_READ_10M_100M,
            POSIX_SIZE_READ_100M_1G, POSIX_SIZE_READ_1G_PLUS, POSIX_SIZE_WRITE_0_100, POSIX_SIZE_WRITE_100_1K,
            POSIX_SIZE_WRITE_1K_10K, POSIX_SIZE_WRITE_10K_100K, POSIX_SIZE_WRITE_100K_1M, POSIX_SIZE_WRITE_1M_4M,
            POSIX_SIZE_WRITE_4M_10M, POSIX_SIZE_WRITE_10M_100M, POSIX_SIZE_WRITE_100M_1G, POSIX_SIZE_WRITE_1G_PLUS)),
        columns=perc_features2)
    return mydata


def extracting_throught(path):
    agg_perf_by_slowest = []
    AGG_PERF_BY_SLOWEST_PATTERN = '(#\s+agg_perf_by_slowest:\s+)(\d+\.\d+)(\s+#+\s)(\S+)'
    ds_agg_perf_by_slowest_pattern = re.compile(AGG_PERF_BY_SLOWEST_PATTERN)

    darshan_files = glob.glob(path)

    for file_name in darshan_files:
        with open(file_name) as infile:
            for line in infile:
                # if line == '# MPI-IO module data\n':
                #    break
                agg_perf_by_slowest_match = ds_agg_perf_by_slowest_pattern.match(line)
                if agg_perf_by_slowest_match is not None:
                    agg_perf_by_slowest_ = float(agg_perf_by_slowest_match.group(2))
                    if agg_perf_by_slowest_match.group(4) == 'MiB/s':
                        agg_perf_by_slowest_ = agg_perf_by_slowest_ * 1024 * 1024;
                    elif agg_perf_by_slowest_match.group(4) == 'KiB/s':
                        agg_perf_by_slowest_ = agg_perf_by_slowest_ * 1024;
                    elif agg_perf_by_slowest_match.group(4) == 'GiB/s':
                        agg_perf_by_slowest_ = agg_perf_by_slowest_ * 1024 * 1024 * 1024;

                    agg_perf_by_slowest.append(agg_perf_by_slowest_)
                    break

    mydata = pd.DataFrame(list(
        zip(agg_perf_by_slowest)),
        columns=['agg_perf_by_slowest'])
    return mydata


def extracting_romio(path):
    cb_read = []
    cb_write = []
    ds_read = []
    ds_write = []
    cb_nodes = []
    cb_config_list = []
    stripe_size = []
    stripe_count = []
    gkfs_chunksize = []
    gkfs_dirents_buff_size = []
    gkfs_daemon_io_xstreams = []
    gkfs_daemon_handler_xstreams = []

    darshan_files = glob.glob(path)

    for file_name in darshan_files:
        last_12 = file_name.split('_')[-12:]
        last_12[-1] = last_12[-1].replace('.txt', '')
        cb_read.append(last_12[0])
        cb_write.append(last_12[1])
        ds_read.append(last_12[2])
        ds_write.append(last_12[3])
        cb_nodes.append(last_12[4])
        cb_config_list.append(last_12[5])
        stripe_size.append(last_12[6])
        stripe_count.append(last_12[7])
        gkfs_chunksize.append(last_12[8])
        gkfs_dirents_buff_size.append(last_12[9])
        gkfs_daemon_io_xstreams.append(last_12[10])
        gkfs_daemon_handler_xstreams.append(last_12[11])

    mydata = pd.DataFrame(list(
        zip(cb_read, cb_write, ds_read, ds_write, cb_nodes, cb_config_list, stripe_size, stripe_count, gkfs_chunksize,
            gkfs_dirents_buff_size, gkfs_daemon_io_xstreams, gkfs_daemon_handler_xstreams)),
        columns=['cb_read', 'cb_write', 'ds_read', 'ds_write', 'cb_nodes', 'cb_config_list', 'stripe_size',
                 'stripe_count', 'gkfs_chunksize', 'gkfs_dirents_buff_size', 'gkfs_daemon_io_xstreams',
                 'gkfs_daemon_handler_xstreams'])
    return mydata


def extracting_darshan(path):
    df1 = extracting_log(path)
    df2 = extracting_perc1(path)
    df3 = extracting_perc2(path)
    df4 = extracting_throught(path)
    df5 = extracting_romio(path)
    df = pd.concat([df1, df2, df3, df5, df4], axis=1)
    df = convert_POSIX_features_to_percentages(df)
    df = log_scale_dataset(df)
    return df


def extracting_darshan57(path):
    """
    从darshan日志文件中提取57个特征
    """
    # 1. 提取基本特征
    df1 = extracting_log(path)
    df2 = extracting_perc1(path)
    df3 = extracting_perc2(path)
    df4 = extracting_throught(path)

    # 2. 合并所有特征
    df = pd.concat([df1, df2, df3, df4], axis=1)

    # 3. 保存NODES列
    nodes_data = None
    if 'NODES' in df.columns:
        nodes_data = df['NODES'].copy()
        df = df.drop(columns=['NODES'])

    # 4. 转换为百分比特征
    df = convert_POSIX_features_to_percentages(df)

    # 5. 对数值特征进行log10转换
    df = log_scale_dataset(df)

    # 6. 重新添加NODES列
    if nodes_data is not None:
        df['NODES'] = nodes_data

    # 7. 确保NODES列在正确的位置
    if 'NODES' in df.columns:
        cols = ['NODES'] + [col for col in df.columns if col != 'NODES']
        df = df[cols]

    return df


if __name__ == "__main__":
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)

    path = sys.argv[1]
    datapath = sys.argv[2]
    df = extracting_darshan(path + '/*.txt')

    # column_names = list(df.columns)
    # for index, row in df.iterrows():
    #     for column in df.columns:
    #         print(column, ":", row[column])

    df.to_csv(datapath, index=False)
    cmd = "rm -rf " + path
    subprocess.run(cmd, shell=True, capture_output=False, text=True)


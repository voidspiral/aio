import os
import re
import math
import logging
import subprocess
import sys
import numpy as np
import pandas as pd
import glob
import configparser
import shutil
from model_training.feature import *
from typing import List, Dict, Any, Optional, Tuple

from tuning.utils.get57features import extracting_darshan57

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



class FeatureProcessor:
    """处理Darshan日志文件并提取特征的类"""

    @staticmethod
    def _load_config(config_path: Optional[str]) -> configparser.ConfigParser:
        """加载配置文件"""
        config = configparser.ConfigParser()
        if config_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(root_dir, "config", "storage.ini")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config.read(config_path)
        return config

    @staticmethod
    def _setup_environment(config: configparser.ConfigParser):
        """设置环境变量"""
        try:
            # 设置PATH环境变量
            darshan_utils_path = config.get('darshan_path', 'darshan_utils')
            os.environ["PATH"] = darshan_utils_path + ":" + os.environ["PATH"]

            # 设置LD_LIBRARY_PATH环境变量
            darshan_lib_path = config.get('darshan_path', 'darshan_ld_library')
            os.environ["LD_LIBRARY_PATH"] = darshan_lib_path + ":" + os.environ["LD_LIBRARY_PATH"]

            logging.info("Environment variables set successfully")
        except Exception as e:
            logging.error(f"Error setting environment variables: {e}")
            raise

    @staticmethod
    def _normalize_path(path: str) -> str:
        """规范化路径"""
        if path.endswith('/'):
            path = path[:-1]
        return os.path.normpath(path)

    @staticmethod
    def parse_darshan_logs(input_path: str, output_path: str) -> None:
        """解析指定目录下的所有Darshan日志文件"""
        global temp_dir
        try:
            # 规范化路径
            input_path = FeatureProcessor._normalize_path(input_path)
            output_path = FeatureProcessor._normalize_path(output_path)

            # 创建临时目录
            temp_dir = os.path.join(output_path, "temp_parsed_logs")
            os.makedirs(temp_dir, exist_ok=True)

            # 获取输入目录中的所有文件
            contents = os.listdir(input_path)

            # 遍历并处理每个日志文件
            for item in contents:
                try:
                    # 构建输入和输出文件路径
                    input_file = os.path.join(input_path, item)
                    temp_output_file = os.path.join(temp_dir, f"{item[:-8]}.txt")

                    # 构建并执行解析命令
                    command = f"darshan-parser --total --file --perf {input_file}"
                    with open(temp_output_file, "w") as outfile:
                        result = subprocess.run(command, shell=True, stdout=outfile, stderr=subprocess.PIPE, text=True)

                        if result.returncode != 0:
                            logging.error(f"Error parsing {item}: {result.stderr}")
                            continue

                    logging.info(f"Successfully parsed {item}")

                except Exception as e:
                    logging.error(f"Error processing {item}: {e}")
                    continue

            # 提取特征并保存到CSV
            df = extracting_darshan57(os.path.join(temp_dir, "*.txt"))
            features_file = os.path.join(output_path, "data.csv")
            df.to_csv(features_file, index=False)
            logging.info(f"Features extracted and saved to {features_file}")

            # 删除临时目录及其内容
            try:
                shutil.rmtree(temp_dir)
                logging.info("Temporary directory cleaned up")
            except Exception as e:
                logging.error(f"Error cleaning up temporary directory: {e}")

        except Exception as e:
            logging.error(f"Error in parse_darshan_logs: {e}")
            # 确保在发生错误时也清理临时目录
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logging.info("Temporary directory cleaned up after error")
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up temporary directory after error: {cleanup_error}")
            raise

    @staticmethod
    def extract_features(log_path: str) -> pd.DataFrame:
        """从解析后的日志中提取特征"""
        try:
            # 获取所有解析后的文件
            txt_files = glob.glob(log_path)
            if not txt_files:
                logging.error(f"No parsed files found in {log_path}")
                return pd.DataFrame()

            # 提取不同类型的特征
            df_log = FeatureProcessor._extract_log_features(log_path)
            df_perc1 = FeatureProcessor._extract_perc_features1(log_path)
            df_perc2 = FeatureProcessor._extract_perc_features2(log_path)
            df_throughput = FeatureProcessor._extract_throughput_features(log_path)
            df_romio = FeatureProcessor._extract_romio_features(log_path)

            # 合并所有特征
            df = pd.concat([df_log, df_perc1, df_perc2, df_romio, df_throughput], axis=1)

            # 转换为百分比特征
            df = FeatureProcessor._convert_to_percentages(df)

            # 对数值特征进行对数转换
            df = FeatureProcessor._log_scale_dataset(df)

            return df

        except Exception as e:
            logging.error(f"Error in extract_features: {e}")
            raise

    @staticmethod
    def _extract_log_features(log_path: str) -> pd.DataFrame:
        """提取基本日志特征 """
        # 初始化所有特征列表
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

        # 正则表达式模式
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

        darshan_files = glob.glob(log_path)

        for file_name in darshan_files:
            # 提取节点规模
            nodes_match = ds_nodes_pattern.search(file_name)
            if nodes_match is not None:
                nodes = int(nodes_match.group(1))
                NODES.append(nodes)
            else:
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
                        POSIX_UNIQUE_FILES.append(posix_unique_files)
                        continue

                    shared_files_match = ds_shared_files_pattern.match(line)
                    if shared_files_match is not None:
                        posix_shared_files = int(shared_files_match.group(2))
                        POSIX_SHARED_FILES.append(posix_shared_files)
                        continue

        POSIX_TOTAL_ACCESSES = [sum(x) for x in zip(POSIX_READS, POSIX_WRITES)]
        POSIX_TOTAL_BYTES = [sum(x) for x in zip(POSIX_BYTES_READ, POSIX_BYTES_WRITTEN)]
        POSIX_TOTAL_FILES = [sum(x) for x in zip(POSIX_SHARED_FILES, POSIX_UNIQUE_FILES)]

        mydata = pd.DataFrame(list(
            zip(POSIX_OPENS, POSIX_SEEKS, POSIX_STATS, POSIX_MMAPS, POSIX_FSYNCS, POSIX_MODE, POSIX_MEM_ALIGNMENT,
                POSIX_FILE_ALIGNMENT, NPROCS, POSIX_TOTAL_ACCESSES, POSIX_TOTAL_BYTES, POSIX_TOTAL_FILES, NODES)),
            columns=log_features)
        return mydata

    @staticmethod
    def _extract_perc_features1(log_path: str) -> pd.DataFrame:
        """提取第一组百分比特征"""
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

        darshan_files = glob.glob(log_path)
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
                POSIX_MEM_NOT_ALIGNED, POSIX_ACCESS1_COUNT, POSIX_ACCESS2_COUNT, POSIX_ACCESS3_COUNT,
                POSIX_ACCESS4_COUNT)),
            columns=perc_features1)
        return mydata

    @staticmethod
    def _extract_perc_features2(log_path: str) -> pd.DataFrame:
        """提取第二组百分比特征（文件大小分布）"""
        features = {feature: [] for feature in perc_features2}

        # 正则表达式模式
        patterns = {
            feature: f'total_{feature}:\s+(\d+)' for feature in perc_features2
        }

        for file_name in glob.glob(log_path):
            with open(file_name) as infile:
                for line in infile:
                    if line == '# MPI-IO module data\n':
                        break
                    for feature, pattern in patterns.items():
                        match = re.search(pattern, line)
                        if match:
                            features[feature].append(int(match.group(1)))
                            break

        return pd.DataFrame(features)

    @staticmethod
    def _extract_throughput_features(log_path: str) -> pd.DataFrame:
        """提取吞吐量特征"""
        throughput = []

        pattern = r'#\s+agg_perf_by_slowest:\s+(\d+\.\d+)\s+#+\s(\S+)'

        for file_name in glob.glob(log_path):
            with open(file_name) as infile:
                for line in infile:
                    match = re.search(pattern, line)
                    if match:
                        value = float(match.group(1))
                        unit = match.group(2)

                        # 转换为字节/秒
                        if unit == 'MiB/s':
                            value *= 1024 * 1024
                        elif unit == 'KiB/s':
                            value *= 1024
                        elif unit == 'GiB/s':
                            value *= 1024 * 1024 * 1024

                        throughput.append(value)
                        break

        return pd.DataFrame({'agg_perf_by_slowest': throughput})

    @staticmethod
    def _extract_romio_features(log_path: str) -> pd.DataFrame:
        """提取ROMIO配置特征"""
        features = {feature: [] for feature in romio_features + lustre_features + gkfs_features}

        for file_name in glob.glob(log_path):
            # 从文件名中提取特征
            last_12 = file_name.split('_')[-12:]
            last_12[-1] = last_12[-1].replace('.txt', '')

            for feature, value in zip(features.keys(), last_12):
                features[feature].append(value)

        return pd.DataFrame(features)

    @staticmethod
    def _convert_to_percentages(df: pd.DataFrame) -> pd.DataFrame:
        """将特征转换为百分比"""
        df = df.copy()

        # 保存NODES列
        nodes_data = None
        if 'NODES' in df.columns:
            nodes_data = df['NODES'].copy()
            df = df.drop(columns=['NODES'])

        # 计算总数
        total_accesses = df['POSIX_TOTAL_ACCESSES']
        total_bytes = df['POSIX_TOTAL_BYTES']
        total_files = df['POSIX_TOTAL_FILES']

        # 转换为百分比
        try:
            # 字节相关特征
            df['POSIX_BYTES_READ_PERC'] = df['POSIX_BYTES_READ'] / total_bytes
            df['POSIX_UNIQUE_BYTES_PERC'] = df['POSIX_UNIQUE_BYTES'] / total_bytes
            df['POSIX_SHARED_BYTES_PERC'] = df['POSIX_SHARED_BYTES'] / total_bytes
            df['POSIX_READ_ONLY_BYTES_PERC'] = df['POSIX_READ_ONLY_BYTES'] / total_bytes
            df['POSIX_READ_WRITE_BYTES_PERC'] = df['POSIX_READ_WRITE_BYTES'] / total_bytes
            df['POSIX_WRITE_ONLY_BYTES_PERC'] = df['POSIX_WRITE_ONLY_BYTES'] / total_bytes

            # 文件相关特征
            df['POSIX_UNIQUE_FILES_PERC'] = df['POSIX_UNIQUE_FILES'] / total_files
            df['POSIX_SHARED_FILES_PERC'] = df['POSIX_SHARED_FILES'] / total_files
            df['POSIX_READ_ONLY_FILES_PERC'] = df['POSIX_READ_ONLY_FILES'] / total_files
            df['POSIX_READ_WRITE_FILES_PERC'] = df['POSIX_READ_WRITE_FILES'] / total_files
            df['POSIX_WRITE_ONLY_FILES_PERC'] = df['POSIX_WRITE_ONLY_FILES'] / total_files

            # 访问相关特征
            df['POSIX_READS_PERC'] = df['POSIX_READS'] / total_accesses
            df['POSIX_WRITES_PERC'] = df['POSIX_WRITES'] / total_accesses
            df['POSIX_RW_SWITCHES_PERC'] = df['POSIX_RW_SWITCHES'] / total_accesses
            df['POSIX_SEQ_READS_PERC'] = df['POSIX_SEQ_READS'] / total_accesses
            df['POSIX_SEQ_WRITES_PERC'] = df['POSIX_SEQ_WRITES'] / total_accesses
            df['POSIX_CONSEC_READS_PERC'] = df['POSIX_CONSEC_READS'] / total_accesses
            df['POSIX_CONSEC_WRITES_PERC'] = df['POSIX_CONSEC_WRITES'] / total_accesses
            df['POSIX_FILE_NOT_ALIGNED_PERC'] = df['POSIX_FILE_NOT_ALIGNED'] / total_accesses
            df['POSIX_MEM_NOT_ALIGNED_PERC'] = df['POSIX_MEM_NOT_ALIGNED'] / total_accesses

            # 文件大小分布特征
            for feature in perc_features2:
                df[f'{feature}_PERC'] = df[feature] / total_accesses

            # 访问计数特征
            df['POSIX_ACCESS1_COUNT_PERC'] = df['POSIX_ACCESS1_COUNT'] / total_accesses
            df['POSIX_ACCESS2_COUNT_PERC'] = df['POSIX_ACCESS2_COUNT'] / total_accesses
            df['POSIX_ACCESS3_COUNT_PERC'] = df['POSIX_ACCESS3_COUNT'] / total_accesses
            df['POSIX_ACCESS4_COUNT_PERC'] = df['POSIX_ACCESS4_COUNT'] / total_accesses

        except Exception as e:
            logging.error(f"Error converting features to percentages: {e}")
            raise

        # 重新添加NODES列
        if nodes_data is not None:
            df['NODES'] = nodes_data

        return df

    @staticmethod
    def _log_scale_dataset(df: pd.DataFrame, add_small_value: float = 1.1,
                           set_NaNs_to: float = -10) -> pd.DataFrame:
        """对数据集进行对数转换"""
        df = df.copy()

        # 保存NODES列
        nodes_data = None
        if 'NODES' in df.columns:
            nodes_data = df['NODES'].copy()
            df = df.drop(columns=['NODES'])

        # 对数值列进行对数转换
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if 'perc' not in column.lower():
                df[f'{column}_LOG10'] = np.log10(df[column] + add_small_value).fillna(value=set_NaNs_to)
                df = df.drop(columns=[column])

        # 重新添加NODES列
        if nodes_data is not None:
            df['NODES'] = nodes_data

        return df

    @staticmethod
    def extracting_darshan57(log_path: str) -> pd.DataFrame:
        """
        从darshan日志文件中提取57个特征 不包含romio的特征

        Args:
            log_path: 日志文件路径

        Returns:
            pd.DataFrame: 包含57个特征的DataFrame
        """
        try:
            # 1. 提取基本特征
            df_log = FeatureProcessor._extract_log_features(log_path)
            df_perc1 = FeatureProcessor._extract_perc_features1(log_path)
            df_perc2 = FeatureProcessor._extract_perc_features2(log_path)
            df_throughput = FeatureProcessor._extract_throughput_features(log_path)

            # 2. 合并所有特征
            df = pd.concat([df_log, df_perc1, df_perc2, df_throughput], axis=1)

            # 3. 保存NODES列
            nodes_data = None
            if 'NODES' in df.columns:
                nodes_data = df['NODES'].copy()
                df = df.drop(columns=['NODES'])

            # 4. 转换为百分比特征
            df = FeatureProcessor._convert_to_percentages(df)

            # 5. 对数值特征进行log10转换
            df = FeatureProcessor._log_scale_dataset(df)

            # 6. 重新添加NODES列
            if nodes_data is not None:
                df['NODES'] = nodes_data

            # 7. 确保NODES列在正确的位置
            if 'NODES' in df.columns:
                cols = ['NODES'] + [col for col in df.columns if col != 'NODES']
                df = df[cols]

            return df

        except Exception as e:
            logging.error(f"Error in extracting_darshan57: {e}")
            raise


def main():

    # 加载配置
    config = FeatureProcessor._load_config(None)
    FeatureProcessor._setup_environment(config)

    # 解析所有日志文件
    input_path = "/thfs3/home/wuhuijun/lmj/fix/AIO_OPRAEL-20250219/dataset_gekkofs"
    output_path = "./model_training/"
    FeatureProcessor.parse_darshan_logs(input_path, output_path)

    input_path = "/thfs3/home/wuhuijun/lmj/fix/AIO_OPRAEL-20250219/dataset_lustre"
    output_path = "./model_training/"
    FeatureProcessor.parse_darshan_logs(input_path, output_path)



if __name__ == "__main__":
    main()
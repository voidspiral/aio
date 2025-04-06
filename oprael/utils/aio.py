log_features = [
    'POSIX_OPENS_LOG10', 'POSIX_SEEKS_LOG10', 'POSIX_STATS_LOG10',
    'POSIX_MMAPS_LOG10', 'POSIX_FSYNCS_LOG10',
    'POSIX_MODE_LOG10', 'POSIX_MEM_ALIGNMENT_LOG10',
    'POSIX_FILE_ALIGNMENT_LOG10', 'NPROCS_LOG10',
    'POSIX_TOTAL_ACCESSES_LOG10', 'POSIX_TOTAL_BYTES_LOG10',
    'POSIX_TOTAL_FILES_LOG10', 'NODES'
]
perc_features = [
    'POSIX_BYTES_READ_PERC', 'POSIX_UNIQUE_BYTES_PERC', 'POSIX_SHARED_BYTES_PERC',
    'POSIX_READ_ONLY_BYTES_PERC', 'POSIX_READ_WRITE_BYTES_PERC',
    'POSIX_WRITE_ONLY_BYTES_PERC', 'POSIX_UNIQUE_FILES_PERC',
    'POSIX_SHARED_FILES_PERC', 'POSIX_READ_ONLY_FILES_PERC',
    'POSIX_READ_WRITE_FILES_PERC', 'POSIX_WRITE_ONLY_FILES_PERC',
    'POSIX_READS_PERC', 'POSIX_WRITES_PERC', 'POSIX_RW_SWITCHES_PERC',
    'POSIX_SEQ_READS_PERC', 'POSIX_SEQ_WRITES_PERC', 'POSIX_CONSEC_READS_PERC',
    'POSIX_CONSEC_WRITES_PERC', 'POSIX_FILE_NOT_ALIGNED_PERC',
    'POSIX_MEM_NOT_ALIGNED_PERC', 'POSIX_SIZE_READ_0_100_PERC',
    'POSIX_SIZE_READ_100_1K_PERC', 'POSIX_SIZE_READ_1K_10K_PERC',
    'POSIX_SIZE_READ_10K_100K_PERC', 'POSIX_SIZE_READ_100K_1M_PERC',
    'POSIX_SIZE_READ_1M_4M_PERC', 'POSIX_SIZE_READ_4M_10M_PERC',
    'POSIX_SIZE_READ_10M_100M_PERC', 'POSIX_SIZE_READ_100M_1G_PERC',
    'POSIX_SIZE_READ_1G_PLUS_PERC', 'POSIX_SIZE_WRITE_0_100_PERC',
    'POSIX_SIZE_WRITE_100_1K_PERC', 'POSIX_SIZE_WRITE_1K_10K_PERC',
    'POSIX_SIZE_WRITE_10K_100K_PERC', 'POSIX_SIZE_WRITE_100K_1M_PERC',
    'POSIX_SIZE_WRITE_1M_4M_PERC', 'POSIX_SIZE_WRITE_4M_10M_PERC',
    'POSIX_SIZE_WRITE_10M_100M_PERC', 'POSIX_SIZE_WRITE_100M_1G_PERC',
    'POSIX_SIZE_WRITE_1G_PLUS_PERC', 'POSIX_ACCESS1_COUNT_PERC',
    'POSIX_ACCESS2_COUNT_PERC', 'POSIX_ACCESS3_COUNT_PERC',
    'POSIX_ACCESS4_COUNT_PERC'
]
#romio_features = ['Romio_CB_Read', 'Romio_CB_Write', 'Romio_DS_Read', 'Romio_DS_Write', 'Cb_nodes', 'Cb_config']
#lustre_feature = ['Strip_Size', 'Strip_Count']
romio_features = ['cb_read', 'cb_write', 'ds_read', 'ds_write', 'cb_nodes', 'cb_config_list']
lustre_feature = ['stripe_size', 'stripe_count']



gkfs_feature = ['gkfs_chunksize','gkfs_dirents_buff_size','gkfs_daemon_io_xstreams','gkfs_daemon_handler_xstreams']
import configparser
import os

def load_config():
    config = configparser.ConfigParser()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(root_dir, "config", "storage.ini")
    config.read(config_path)
    return config

if __name__ == '__main__':
    load_config()
# License: MIT

import numpy as np
from oprael import space as sp, OPRAELOptimizer
import argparse
import os
import subprocess
import sys
import joblib
import pandas as pd
from oprael.utils.aio import log_features, perc_features, romio_features, lustre_feature
from tuning.utils.get57features import extracting_darshan57
import configparser
from pathlib import Path

'''
argparser = argparse.ArgumentParser()
argparser.add_argument('--access',default="w",help="r replace only read,w replace only write,rw replace read and write")
argparser.add_argument("--MPIN",type=str,default="8",help="The MPI Node to run the benchmark.")
argparser.add_argument("--process",type=str,default="64",help="The number of process to run the benchmark.")
args = argparser.parse_args()
'''

# Define Search Space
space_lustre = sp.Space()
strp_fac = sp.Int("stripe_count", 1, 64, default_value=1)
strp_unt = sp.Int("stripe_size", 1, 128, default_value=1)
rm_cb_read = sp.Int("cb_read", 0, 2, default_value=0)
rm_cb_write = sp.Int("cb_write", 0, 2, default_value=0)
rm_ds_read = sp.Int("ds_read", 0, 2, default_value=0)
rm_ds_write = sp.Int("ds_write", 0, 2, default_value=0)
cb_nodes = sp.Int("cb_nodes", 1, 32, default_value=4)
config_list = sp.Int("cb_config_list", 1, 32, default_value=1)
space_lustre.add_variables(
    [strp_fac, strp_unt, rm_cb_read, rm_cb_write, rm_ds_read, rm_ds_write, cb_nodes, config_list])

space_gekkofs = sp.Space()
gkfs_chunksize = sp.Int("gkfs_chunksize", 1, 16, default_value=1)  # mb
gkfs_dirents_buff_size = sp.Int("gkfs_dirents_buff_size", 1, 16, default_value=1)  # MB
gkfs_daemon_io_xstreams = sp.Int("gkfs_daemon_io_xstreams", 4, 32, default_value=4)
gkfs_daemon_handler_xstreams = sp.Int("gkfs_daemon_handler_xstreams", 2, 16, default_value=2)
space_gekkofs.add_variables(
    [gkfs_chunksize, gkfs_dirents_buff_size, gkfs_daemon_io_xstreams, gkfs_daemon_handler_xstreams])

Mbytes = 1024 * 1024
states_mapping = ["automatic", "disable", "enable"]


def getIO(path):
    with open(path, "r") as f:
        contents = f.readlines()
    performance = -10000

    for _ in contents:
        if "write bandwidth" in _:
            performance = float(_.split(":")[1].replace("MiB/s", "").strip())
    return performance


def load_config():
    config = configparser.ConfigParser()
    root_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(root_dir, "config", "storage.ini")
    config.read(config_path)
    return config


def update_gekkofs_env(config: dict) -> None:
    """更新 GekkoFS 环境变量配置文件

    Args:
        config: 包含 GekkoFS 配置参数的字典
    """
    try:
        chunk_size = config["gkfs_chunksize"] * 1024 * 1024
        dirents_buff_size = config["gkfs_dirents_buff_size"] * 1024 * 1024
        daemon_io_streams = config["gkfs_daemon_io_xstreams"]
        daemon_handler_streams = config["gkfs_daemon_handler_xstreams"]

        configs = configparser.ConfigParser()
        root_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(root_dir, "config", "storage.ini")

        configs.read(config_path)
        aio_path = os.path.join(configs.get('gekkofs_path', 'gekkofs_home'), 'scripts/yh-run/aio.sh')

        with open(aio_path, 'r') as file:
            lines = file.readlines()

        # 更新配置值
        new_lines = []
        for line in lines:
            if 'export GKFS_RPC_CHUNKSIZE=' in line:
                new_lines.append(f'export GKFS_RPC_CHUNKSIZE={chunk_size}\n')
            elif 'export GKFS_RPC_DIRENTS_BUFF_SIZE=' in line:
                new_lines.append(f'export GKFS_RPC_DIRENTS_BUFF_SIZE={dirents_buff_size}\n')
            elif 'export GKFS_RPC_DAEMON_IO_XSTREAMS=' in line:
                new_lines.append(f'export GKFS_RPC_DAEMON_IO_XSTREAMS={daemon_io_streams}\n')
            elif 'export GKFS_RPC_DAEMON_HANDLER_XSTREAMS=' in line:
                new_lines.append(f'export GKFS_RPC_DAEMON_HANDLER_XSTREAMS={daemon_handler_streams}\n')
            else:
                new_lines.append(line)

        with open(aio_path, 'w') as file:
            file.writelines(new_lines)

        os.environ["GKFS_RPC_CHUNKSIZE"] = str(chunk_size)
        os.environ["GKFS_RPC_DIRENTS_BUFF_SIZE"] = str(dirents_buff_size)
        os.environ["GKFS_RPC_DAEMON_IO_XSTREAMS"] = str(daemon_io_streams)
        os.environ["GKFS_RPC_DAEMON_HANDLER_XSTREAMS"] = str(daemon_handler_streams)

        print(f"Updated GekkoFS config with: chunk_size={chunk_size}B, buffer_size={dirents_buff_size}B, "
              f"io_streams={daemon_io_streams}, handler_streams={daemon_handler_streams}")

    except Exception as e:
        print(f"Error updating GekkoFS config: {e}")
        raise


# Define Objective Function
def eval_func(config, fs_type, cmd_para):
    cmd_para = global_command_para
    if fs_type == 'Lustre':
        try:
            this_strp_fac = config["stripe_count"]
            this_strp_uni = config["stripe_size"] * Mbytes
            this_rm_cb_read = config["cb_read"]
            this_rm_cb_write = config["cb_write"]
            this_ds_read = config["ds_read"]
            this_ds_write = config["ds_write"]
            this_cb_nodes = config["cb_nodes"]
            this_config_list = config["cb_config_list"]

            print("Evaluate Parameters config (%d, %d, %d, %d, %d, %d, %d, %d):" % (
                this_strp_fac, this_strp_uni, this_rm_cb_read, this_rm_cb_write,
                this_ds_read, this_ds_write, this_cb_nodes, this_config_list))

            configs = load_config()
            hint = configs.get('tuning_path', 'hint')
            with open(hint, 'w') as f:
                f.write(str(this_rm_cb_read))
                f.write('\n')
                f.write(str(this_rm_cb_write))
                f.write('\n')
                f.write(str(this_ds_read))
                f.write('\n')
                f.write(str(this_ds_write))
                f.write('\n')
                f.write(str(this_cb_nodes))
                f.write('\n')
                f.write(str(this_config_list))

            this_stripe_size = this_strp_uni
            this_stripe_count = this_strp_fac

            test_dir = os.path.join(configs.get('tuning_path', 'optfiles'))
            command = f"lfs setstripe -S {this_stripe_size} -c {this_stripe_count} {test_dir}"
            print(command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error running command. Return code: {result.returncode}")
                print(f"Error output: {result.stderr}")
                return {'objectives': [10000]}

            os.environ["LD_LIBRARY_PATH"] = configs.get('darshan_path', 'darshan_ld_library') + ":" + os.environ[
                "LD_LIBRARY_PATH"]
            os.environ["PATH"] = configs.get('darshan_path', 'darshan_utils') + ":" + os.environ["PATH"]
            os.environ["LD_PRELOAD"] = configs.get('darshan_path', 'darshan_runtime')
            tuning_files = os.path.join(configs.get("root_path", "home"), "tuningFiles")
            Path(tuning_files).mkdir(parents=True, exist_ok=True)

            darshan_file = f'{tuning_files}/{this_strp_fac}_{this_strp_uni}_{this_rm_cb_read}_{this_rm_cb_write}_{this_ds_read}_{this_ds_write}_{this_cb_nodes}_{this_config_list}.darshan'
            txt_file = f'{tuning_files}/{this_strp_fac}_{this_strp_uni}_{this_rm_cb_read}_{this_rm_cb_write}_{this_ds_read}_{this_ds_write}_{this_cb_nodes}_{this_config_list}.txt'
            os.environ["DARSHAN_LOGFILE"] = darshan_file

            result = subprocess.run(cmd_para, shell=True, capture_output=True, text=True)
            print(cmd_para)
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error running benchmark command. Return code: {result.returncode}")
                print(f"Error output: {result.stderr}")
                return {'objectives': [10000]}

            parser_cmd = f'darshan-parser --show-incomplete --base --perf {darshan_file} >> {txt_file}'
            result = subprocess.run(parser_cmd, shell=True, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"Error parsing darshan file. Return code: {result.returncode}")
                return {'objectives': [10000]}

            with open(txt_file, 'r') as infile:
                for line in infile:
                    if 'agg_perf_by_slowest' in line:
                        line = line.split()
                        performance = float(line[2])
                        break

            record_file = os.path.join(tuning_files, 'record.txt')
            with open(record_file, 'a') as file:
                file.write(f"{performance}\n")

            print(f"Config: {config}, Performance: {performance}")
            return {'objectives': [-performance]}

        except Exception as e:
            print(f"Error in eval_func: {e}")
            return {'objectives': [10000]}

    elif fs_type == 'GekkoFS':
        try:
            # 获取GekkoFS配置参数
            this_gkfs_chunksize = config["gkfs_chunksize"]
            this_gkfs_dirents_buff_size = config["gkfs_dirents_buff_size"]
            this_gkfs_daemon_io_xstreams = config["gkfs_daemon_io_xstreams"]
            this_gkfs_daemon_handler_xstreams = config["gkfs_daemon_handler_xstreams"]

            print("Evaluate GekkoFS Parameters config (%d, %d, %d, %d):" % (
                this_gkfs_chunksize, this_gkfs_dirents_buff_size,
                this_gkfs_daemon_io_xstreams, this_gkfs_daemon_handler_xstreams))

            # 读取配置文件
            configs = load_config()

            # 更新 GekkoFS 系统参数
            update_gekkofs_env(config)

            # 设置GekkoFS环境变量
            os.environ["PKG_CONFIG_PATH"] = configs.get('gekkofs_path', 'gekkofs_home') + configs.get('gekkofs_path',
                                                                                                      'deps_path') + configs.get(
                'gekkofs_path', 'pkg_config_path')
            os.environ["CMAKE_PREFIX_PATH"] = configs.get('gekkofs_path', 'gekkofs_home') + configs.get('gekkofs_path',
                                                                                                        'deps_path')
            os.environ["LD_LIBRARY_PATH"] = configs.get('gekkofs_path', 'gekkofs_home') + configs.get('gekkofs_path',
                                                                                                      'deps_path') + "/lib:" + configs.get(
                'gekkofs_path', 'gekkofs_home') + configs.get('gekkofs_path',
                                                              'deps_path') + "/lib64:/lib/aarch64-linux-gnu:" + \
                                            os.environ["LD_LIBRARY_PATH"]
            os.environ["UCX_TLS"] = configs.get('gekkofs_path', 'ucx_tls')
            os.environ["UCX_NET_DEVICES"] = configs.get('gekkofs_path', 'ucx_net_devices')
            os.environ["LIBGKFS_REGISTRY"] = configs.get('gekkofs_path', 'gekkofs_reg')

            # 写入GekkoFS参数
            hint = os.path.join(configs.get('tuning_path', 'optfiles'), "parameter_gekkofs.txt")
            with open(hint, 'w') as f:
                f.write(str(this_gkfs_chunksize))
                f.write('\n')
                f.write(str(this_gkfs_dirents_buff_size))
                f.write('\n')
                f.write(str(this_gkfs_daemon_io_xstreams))
                f.write('\n')
                f.write(str(this_gkfs_daemon_handler_xstreams))

            # 重启GekkoFS守护进程
            gkfs_home = configs.get("gekkofs_path", "gekkofs_home")
            commands = [
                f'cd {gkfs_home}',
                'cd scripts/yh-run/',
                f'source {gkfs_home}/scripts/yh-run/aio.sh',
                f'./restart.sh {global_nodes_list}'
            ]
            cmd = " && ".join(commands)
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error restarting GekkoFS. Return code: {result.returncode}")
                return {'objectives': [10000]}
            print(result.stdout)
            # 设置Darshan环境
            os.environ["LD_LIBRARY_PATH"] = configs.get('darshan_path', 'darshan_ld_library') + ":" + os.environ[
                "LD_LIBRARY_PATH"]
            os.environ["PATH"] = configs.get('darshan_path', 'darshan_utils') + ":" + os.environ["PATH"]
            os.environ["LD_PRELOAD"] = configs.get('darshan_path', 'darshan_runtime')
            tuning_files = os.path.join(configs.get("root_path", "home"), "tuningFiles")
            Path(tuning_files).mkdir(parents=True, exist_ok=True)

            # 设置Darshan日志文件
            darshan_file = f'{tuning_files}/gkfs_{this_gkfs_chunksize}_{this_gkfs_dirents_buff_size}_{this_gkfs_daemon_io_xstreams}_{this_gkfs_daemon_handler_xstreams}.darshan'
            txt_file = f'{tuning_files}/gkfs_{this_gkfs_chunksize}_{this_gkfs_dirents_buff_size}_{this_gkfs_daemon_io_xstreams}_{this_gkfs_daemon_handler_xstreams}.txt'
            os.environ["DARSHAN_LOGFILE"] = darshan_file

            # 运行基准测试
            # TODO cmd_para -o /dev/shm/gkfs
            # set environ to gkfs client(os.envrion) server(source aio.sh)
            # mpirun command insert -env
            # cmd_para = f"{cmd_para} -env LD_PRELOAD=$client -c -o /dev/shm/gkfs"
            client_lib = os.path.join(configs.get('gekkofs_path', 'gekkofs_home'),
                                      configs.get('gekkofs_path', 'gekkofs_client'))

            # 分割命令参数
            parts = cmd_para.split()

            if parts and parts[0] == "mpirun":
                # 找到 -hosts 参数的位置
                hosts_index = -1
                for i, part in enumerate(parts):
                    if part == "-hosts":
                        hosts_index = i
                        break

                if hosts_index != -1 and hosts_index + 1 < len(parts):
                    # 分割命令为三部分：mpirun到hosts参数、hosts后的参数
                    prefix = parts[:hosts_index + 2]  # 包含 mpirun ... -hosts {hosts}
                    suffix = parts[hosts_index + 2:]  # hosts参数后的所有内容

                    # 重新组合命令
                    cmd_para = f"{' '.join(prefix)} -env LD_PRELOAD={client_lib} {' '.join(suffix)} -o /dev/shm/gkfs/testfile"

                    print(f"修改后的命令: {cmd_para}")
                else:
                    print("警告: 未找到 -hosts 参数或格式不正确")
            else:
                print("警告: 命令不是以 mpirun 开头")
            gkfs_hosts_src = os.path.join(gkfs_home, "scripts/yh-run/gkfs_hosts.txt")
            if not os.path.exists(gkfs_hosts_src):
                raise FileNotFoundError(f"GekkoFS hosts file not found: {gkfs_hosts_src}")

            os.environ["LIBGKFS_HOSTS_FILE"] = gkfs_hosts_src

            result = subprocess.run(cmd_para, shell=True, capture_output=True, text=True)
            print(cmd_para)
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error running benchmark command. Return code: {result.returncode}")
                print(f"Error output: {result.stderr}")
                return {'objectives': [10000]}

            # 解析Darshan日志
            parser_cmd = f'darshan-parser --show-incomplete --base --perf {darshan_file} >> {txt_file}'
            result = subprocess.run(parser_cmd, shell=True, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"Error parsing darshan file. Return code: {result.returncode}")
                return {'objectives': [10000]}

            # 提取性能数据
            with open(txt_file, 'r') as infile:
                for line in infile:
                    if 'agg_perf_by_slowest' in line:
                        line = line.split()
                        performance = float(line[2])
                        break

            # 记录性能数据
            record_file = os.path.join(tuning_files, 'record_gekkofs.txt')
            with open(record_file, 'a') as file:
                file.write(f"{performance}\n")

            print(f"GekkoFS Config: {config}, Performance: {performance}")
            return {'objectives': [-performance]}

        except Exception as e:
            del os.environ["GKFS_RPC_CHUNKSIZE"]
            del os.environ["GKFS_RPC_DIRENTS_BUFF_SIZE"]
            del os.environ["GKFS_RPC_DAEMON_IO_XSTREAMS"]
            del os.environ["GKFS_RPC_DAEMON_HANDLER_XSTREAMS"]

            print(f"Error in GekkoFS eval_func: {e}")
            return {'objectives': [10000]}


def searchBest(cmd, fs_type, command_para, nodes_list, log_file_2=None):
    global global_command_para
    global global_nodes_list

    global_command_para = command_para
    global_nodes_list = nodes_list

    best_config = None
    best_perf = None
    model = None

    configs = load_config()
    task_id = 'Aio ior task'
    opt = OPRAELOptimizer(
        eval_func,
        cmd=cmd,
        log_file_2=log_file_2,
        fs_type=fs_type,
        config_space_lustre=space_lustre,
        config_space_gekkofs=space_gekkofs,
        best_config=best_config,
        best_perf=best_perf,
        model=model,
        config_space=space_gekkofs if fs_type == "GekkoFS" else space_lustre,  # 这里会随着fstype变化
        max_runs=1,
        runtime_limit=600000,
        advisor_type='custom',
        time_limit_per_trial=120,
        task_id=task_id,
        access='w',
        custom_advisor_list=["tpe", "ga", "bo"],
        ori_nodelist=nodes_list
    )

    history = opt.run()[0]
    best_config = opt.run()[1]
    best_perf = np.power(10, np.abs(opt.run()[2])) / 1024 / 1024
    return best_config, best_perf
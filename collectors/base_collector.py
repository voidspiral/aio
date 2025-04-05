import configparser
import os
import subprocess
import time
from abc import ABC, abstractmethod


class BaseCollector(ABC):
    def __init__(self, fs_type, nodelist, output):
        self.fs_type = fs_type
        self.nodelist = nodelist
        self.output = output
        self.config = self.load_config()
        self.root_path = self.config.get('root_path', 'home')

    @staticmethod
    def load_config():
        config = configparser.ConfigParser()
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(root_dir, "config", "storage.ini")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        config.read(config_path)
        return config

    def config_env(self):
        """配置基本环境变量"""
        # MPI环境
        mpi_path = self.config.get('mpi_path', 'mpirun')
        mpi_lib = self.config.get('mpi_path', 'mpilib')

        os.environ["PATH"] = f"{mpi_path}:{os.environ['PATH']}"
        os.environ["LD_LIBRARY_PATH"] = f"{mpi_lib}:{os.environ['LD_LIBRARY_PATH']}"

        os.environ["PATH"] = self.config.get('mpi_path', 'mpirun') + ":" + os.environ["PATH"]
        os.environ["LD_LIBRARY_PATH"] = self.config.get('mpi_path', 'mpilib') + ":" + os.environ["LD_LIBRARY_PATH"]
        # Darshan环境
        os.environ["DARSHAN_LOGPATH"] = str(self.output)
        os.environ["LD_PRELOAD"] = self.config.get('darshan_path', 'darshan_runtime')

        # ROMIO环境
        os.environ["LD_PRELOAD"] = f"{self.config.get('romio_path', 'romio_tuner')}:{os.environ['LD_PRELOAD']}"
        # 影响tuning/mpiio.c的读入
        os.environ["ROMIO_HINT_PATH"] = self.config.get('tuning_path', 'hint')

        if self.fs_type == "Lustre":
            # TODO lustre test dir
            return self.config.get('lustre_path', 'lustre')
        elif self.fs_type == "GekkoFS":
            self._config_gekkofs_env()
            return self.config.get('gekkofs_path', 'gekkofs_mount')

    def _config_gekkofs_env(self):
        """配置GekkoFS特定环境"""
        gekkofs_home = self.config.get('gekkofs_path', 'gekkofs_home')
        deps_path = self.config.get('gekkofs_path', 'deps_path')

        # 更新环境变量，完全匹配原始代码的设置
        os.environ["PKG_CONFIG_PATH"] = gekkofs_home + deps_path + self.config.get('gekkofs_path', 'pkg_config_path')
        os.environ["CMAKE_PREFIX_PATH"] = gekkofs_home + deps_path
        os.environ["LD_LIBRARY_PATH"] = (f"{gekkofs_home}{deps_path}/lib:"
                                         f"{gekkofs_home}{deps_path}/lib64:"
                                         "/lib/aarch64-linux-gnu/:"
                                         f"{os.environ['LD_LIBRARY_PATH']}")
        os.environ["UCX_TLS"] = self.config.get('gekkofs_path', 'ucx_tls')
        os.environ["UCX_NET_DEVICES"] = self.config.get('gekkofs_path', 'ucx_net_devices')
        os.environ["LIBGKFS_REGISTRY"] = self.config.get('gekkofs_path', 'gekkofs_reg')

    @abstractmethod
    def collect_trace(self):
        """收集跟踪数据的抽象方法"""
        pass

    def set_romio(self, romio):
        """设置ROMIO参数"""
        hint_path = self.config.get('tuning_path', 'hint')
        with open(hint_path, 'w') as f:
            for param in romio:
                f.write(str(param) + '\n')

    def run_command(self, command, env=None):
        """执行命令并处理错误"""
        print(command)
        result = subprocess.run(command, shell=True, env=env, capture_output=False, text=True)
        if result.returncode != 0:
            with open("errorcommand.txt", "a") as f:
                f.write(command + "\n")
            print("Command failed:", command)
            return False
        return True

    def update_gekkofs_env(self, config: list) -> None:
        """更新 GekkoFS 环境变量配置文件

        Args:
            config: 包含 GekkoFS 配置参数的字典
        """
        try:
            hash = dict()
            hash[0] = 512
            hash[1] = 1024
            hash[2] = 2048
            hash[3] = 4096
            hash[4] = 8192
            chunk_size = hash[config[0] % 5] * 1024
            dirents_buff_size = config[1] * 1024 * 1024
            daemon_io_streams = config[2]
            daemon_handler_streams = config[3]

            aio_path = os.path.join(self.config.get('gekkofs_path', 'gekkofs_home'), 'scripts/yh-run/aio.sh')

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

            # 设置环境变量保证client 和 server的配置一致
            # 更改到ior_command_build
            # os.environ["GKFS_RPC_CHUNKSIZE"] = str(chunk_size)
            # os.environ["GKFS_RPC_DIRENTS_BUFF_SIZE"] = str(dirents_buff_size)
            # os.environ["GKFS_RPC_DAEMON_IO_XSTREAMS"] = str(daemon_io_streams)
            # os.environ["GKFS_RPC_DAEMON_HANDLER_XSTREAMS"] = str(daemon_handler_streams)
            #
            print(f"Updated GekkoFS config with: chunk_size={chunk_size}B, buffer_size={dirents_buff_size}B, "
                  f"io_streams={daemon_io_streams}, handler_streams={daemon_handler_streams}")

        except Exception as e:
            print(f"Error updating GekkoFS config: {e}")

    def set_fs_parameters(self, dir_, count, config_params, nodes_list=None):
        """设置文件系统参数"""
        if self.fs_type == "Lustre":
            self._set_lustre_stripe(dir_, config_params)
        elif self.fs_type == "GekkoFS":
            self._set_gkfs_parameter(config_params, nodes_list)

    def _set_lustre_stripe(self, path, lustre_config):
        """设置 Lustre 条带化参数"""
        stripe_size = lustre_config[0] * 1024 * 1024  # 转换为字节
        stripe_count = lustre_config[1]
        command = f"lfs setstripe -S {stripe_size} -c {stripe_count} {path}"
        self.run_command(command)

    def _set_gkfs_parameter(self, gkfs_config, nodes_list):
        """设置 GekkoFS 参数"""
        # 更新GekkoFS配置参数
        self.update_gekkofs_env(gkfs_config)

        # 每次设置参数时都重启 GekkoFS
        if nodes_list:
            gekkofs_home = self.config.get('gekkofs_path', 'gekkofs_home')

            self._run_gkfs(gekkofs_home, nodes_list)
        else:
            raise RuntimeError("can not capture nodelist")

    def _run_gkfs(self, gekkofs_home, nodes_list):
        """运行 GekkoFS"""

        # 首先运行 GekkoFS
        commands = [
            f'source {gekkofs_home}/scripts/yh-run/aio.sh',
            f'cd {gekkofs_home}',
            'cd scripts/yh-run/',
            f'./restart.sh {nodes_list}'
        ]
        if not self.run_command(" && ".join(commands)):
            raise RuntimeError("Failed to start GekkoFS")
        time.sleep(0.5)
        # 执行 source 命令 TODO 子shell设置可能不生效
        if not self.run_command(f"source {gekkofs_home}/env.sh"):
            raise RuntimeError("Failed to source GekkoFS environment")

        # aio.sh 需要client 和 server 端同时应用

        # GekkoFS 启动后复制 gkfs_hosts.txt
        gkfs_hosts_src = os.path.join(gekkofs_home, "scripts/yh-run/gkfs_hosts.txt")
        gkfs_hosts_dst = os.path.join(self.root_path, "gkfs_hosts.txt")

        # TODO  /tmp/gkfs_daemon.log
        if not os.path.exists(gkfs_hosts_src):
            raise FileNotFoundError(f"GekkoFS hosts file not found: {gkfs_hosts_src}")

        os.environ["LIBGKFS_HOSTS_FILE"] = gkfs_hosts_dst

        self.run_command(f"cp {gkfs_hosts_src} {gkfs_hosts_dst}")


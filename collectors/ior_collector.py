import random
import os
import copy
from .base_collector import BaseCollector
from ClusterShell.NodeSet import NodeSet
from .utils.config_space import ConfigSpace


class IORCollector(BaseCollector):
    def __init__(self, fs_type, nodelist, output, max_samples=10):
        # 保存初始环境变量状态
        self.initial_environ = copy.deepcopy(os.environ)

        super().__init__(fs_type, nodelist, output)
        # nodes  4, 8. 16 32
        # -ppn 8
        # -b 64m 128m 256m `
        # -t 4    16    64
        # offset 不要了
        # TODO timeout
        self.test_params = {
            'nodes': [4, 8, 16, 32],
            'proc_per_node': [8],
            'blocksize': ["64m", "128m", "512m"],
            'xfersize': ["4m"],
        }

        # 创建ConfigSpace实例
        self.max_samples = max_samples
        self.config_space = ConfigSpace(max_samples=self.max_samples)

    def pick_contiguous_nodes(self, count=4):
        """
        从节点集合中选择指定数量的连续节点
        """
        nodeset = NodeSet(self.nodelist)
        all_nodes = list(nodeset)

        if count >= len(all_nodes):
            return all_nodes

        max_start_index = len(all_nodes) - count + 1
        start_index = random.randrange(max_start_index)
        return all_nodes[start_index:start_index + count]

    def collect_trace(self):
        for N in self.test_params['nodes']:
            test_nodes = self.pick_contiguous_nodes(N)
            print("Test node", test_nodes)
            expanded_nodes = self._expand_nodes(test_nodes)

            for xfersize in self.test_params['xfersize']:
                for blocksize in self.test_params['blocksize']:
                    # TODO 重写 -b -t 的倍数关系
                    if not self._validate_sizes(xfersize, blocksize):
                        continue

                    # 枚举参数空间中的样本
                    for count in range(self.max_samples):
                        # TODO 1.timeout
                        #     2.错误的执行不应该异常退出而是try catch后抛出异常,并删除相应的darshan数据
                        try:
                            # TODO
                            test_dir = self.config_env()
                            self._run_single_test(test_dir, expanded_nodes, N, self.test_params['proc_per_node'][0],
                                                  xfersize,
                                                  blocksize, count)
                        except Exception as e:
                            print(f"Error: {e}")
                            print("some case error")
                            print("continue collecting data")
                        finally:
                            # 每次测试后重置环境变量
                            self._reset_environment()

    def _reset_environment(self):
        """完全重置环境变量到初始状态"""
        # 清空当前环境变量
        os.environ.clear()

        # 恢复初始环境变量
        for key, value in self.initial_environ.items():
            os.environ[key] = value

        print("Environment completely reset to initial state")

    def _run_single_test(self, test_dir, nodes, N, ppn, xfersize, blocksize, count):
        """运行单次测试"""
        # 获取当前配置 - 使用ConfigSpace实例方法
        romio_config = self.config_space.get_romio_config(count)
        lustre_config = self.config_space.get_lustre_config(count)
        gkfs_config = self.config_space.get_gkfs_config(count)

        # 设置文件系统参数
        self.set_fs_parameters(test_dir, count,
                               lustre_config if self.fs_type == "Lustre" else gkfs_config,
                               nodes if self.fs_type == "GekkoFS" else None)

        # 测试 MPI 环境
        # mpi_test_cmd = f"mpirun -np 4 -hosts {nodes} hostname"
        # print("\nTesting MPI environment:")
        # if not self.run_command(mpi_test_cmd):
        #     raise RuntimeError("MPI test failed")
        #
        # 设置日志文件名
        romio_str = '_'.join(map(str, romio_config))
        lustre_str = '_'.join(map(str, lustre_config))
        gkfs_str = '_'.join(map(str, gkfs_config))

        #TODO fix lustre_str 不可能和gkfs_str 同时存在
        log_filename = f"N-{N}_n-{ppn}_m_wr_t-{xfersize}_b-{blocksize}_{romio_str}_{lustre_str}_{gkfs_str}.darshan"
        os.environ["DARSHAN_LOGFILE"] = os.path.join(os.environ["DARSHAN_LOGPATH"], log_filename)

        # 设置参数并运行测试
        self.set_romio(romio_config)

        command = self._build_ior_command(nodes, ppn,
                                          xfersize, blocksize, test_dir, gkfs_config)
        self.run_command(command)

    @staticmethod
    def _expand_nodes(pick_node: list):
        """展开节点列表"""
        nodeset = NodeSet.fromlist(pick_node)
        expanded_nodes = list(nodeset)
        return ','.join(str(node) for node in expanded_nodes)

    def _validate_sizes(self, xfersize, blocksize):
        """验证传输大小和块大小的关系"""
        xfer_bytes = self._convert_size_to_bytes(xfersize)
        block_bytes = self._convert_size_to_bytes(blocksize)
        return block_bytes % xfer_bytes == 0

    @staticmethod
    def _convert_size_to_bytes(size_str):
        """转换大小字符串到字节数"""
        units = {'k': 1024, 'm': 1024 ** 2, 'g': 1024 ** 3}
        number = int(size_str[:-1])
        unit = size_str[-1].lower()
        return number * units[unit]

    def _build_ior_command(self, nodes, proc_per_node, xfersize, blocksize, test_dir, gkfs_config=None):
        """构建IOR命令
        Args:
            nodes: 节点列表
            proc_per_node: 每个节点的进程数
            xfersize: 传输大小
            blocksize: 块大小
            test_dir: 测试目录
            gkfs_config: GekkoFS配置参数 [chunksize, dirents_buff_size, daemon_io_streams, daemon_handler_streams]
        """
        ior_path = os.path.join(self.root_path, "io_apps/mpiior")

        if self.fs_type == "GekkoFS":
            client_lib = os.path.join(self.config.get('gekkofs_path', 'gekkofs_home'),
                                      self.config.get('gekkofs_path', 'gekkofs_client'))

            # 如果提供了GekkoFS配置，构建环境变量设置
            if gkfs_config:
                hash = dict()
                hash[0] = 512
                hash[1] = 1024
                hash[2] = 2048
                hash[3] = 4096
                hash[4] = 8192
                chunk_size = hash[gkfs_config[0] % 5] * 1024
                dirents_buff_size = gkfs_config[1] * 1024 * 1024
                daemon_io_streams = gkfs_config[2]
                daemon_handler_streams = gkfs_config[3]

                env_settings = [
                    f"-env LD_PRELOAD={client_lib}:$LD_PRELOAD",
                    f"-env GKFS_RPC_CHUNKSIZE={chunk_size}",
                    f"-env GKFS_RPC_DIRENTS_BUFF_SIZE={dirents_buff_size}",
                    f"-env GKFS_RPC_DAEMON_IO_XSTREAMS={daemon_io_streams}",
                    f"-env GKFS_RPC_DAEMON_HANDLER_XSTREAMS={daemon_handler_streams}"
                ]
                env_string = " ".join(env_settings)
            else:
                env_string = f"-env LD_PRELOAD={client_lib}:$LD_PRELOAD"
            return (f"timeout 300 mpirun -ppn {proc_per_node} -hosts {nodes} "
                    f"{env_string} "
                    f"{ior_path} -a mpiio -c -t {xfersize} -b {blocksize} "
                    f"-o {os.path.join(test_dir, 'testFile')}")
        else:
            return (f"timeout 300 mpirun -ppn {proc_per_node} -hosts {nodes} "
                    f"{ior_path} -a mpiio -c -t {xfersize} -b {blocksize} "
                    f"-o {os.path.join(test_dir, 'testFile')}")

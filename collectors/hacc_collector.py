from .base_collector import BaseCollector
from ClusterShell.NodeSet import NodeSet
from .utils.config_space import ConfigSpace
import os


class HACCCollector(BaseCollector):
    def __init__(self, fs_type, nodelist, output):
        super().__init__(fs_type, nodelist, output)
        self.test_params = {
            'nodes': [4],
            'nproc': [32],
            'particles': ["1000000"]  # 粒子数量
        }
        self.configs = ConfigSpace.init_config_space()

    def collect_trace(self):
        dir_ = self.config_env()
        expanded_nodes = self._expand_nodes()

        for N in range(len(self.test_params['nodes'])):
            for n in range(N, len(self.test_params['nproc'])):
                for particles in self.test_params['particles']:
                    count = 0
                    sum_configs = 15  # 配置组合数量
                    while sum_configs > 0:
                        self._run_single_test(dir_, expanded_nodes, N, n, particles, count)
                        count += 1
                        sum_configs -= 1

    def _run_single_test(self, dir_, nodes, N, n, particles, count):
        """运行单次测试"""
        # 获取当前配置
        romio_config = ConfigSpace.get_romio_config(self.configs, count)
        lustre_config = ConfigSpace.get_lustre_config(self.configs, count)
        gkfs_config = ConfigSpace.get_gkfs_config(self.configs, count)

        # 设置文件系统参数
        self.set_fs_parameters(dir_, count,
                               lustre_config if self.fs_type == "Lustre" else gkfs_config,
                               nodes if self.fs_type == "GekkoFS" else None)

        # 测试 MPI 环境
        mpi_test_cmd = f"mpirun -np 4 -hosts {nodes} hostname"
        print("\nTesting MPI environment:")
        if not self.run_command(mpi_test_cmd):
            raise RuntimeError("MPI test failed")

        # 设置日志文件名
        romio_str = '_'.join(map(str, romio_config))
        lustre_str = '_'.join(map(str, lustre_config))
        gkfs_str = '_'.join(map(str, gkfs_config))

        log_filename = f"N-{self.test_params['nodes'][N]}_n-{self.test_params['nproc'][n]}_p-{particles}_{romio_str}_{lustre_str}_{gkfs_str}.darshan"
        os.environ["DARSHAN_LOGFILE"] = os.path.join(os.environ["DARSHAN_LOGPATH"], log_filename)

        # 设置参数并运行测试
        self.set_romio(romio_config)
        command = self._build_hacc_command(nodes, self.test_params['nproc'][n],
                                           particles, dir_)
        self.run_command(command)

    def _expand_nodes(self):
        """展开节点列表"""
        nodeset = NodeSet(self.nodelist)
        expanded_nodes = list(nodeset)
        return ','.join(str(node) for node in expanded_nodes)

    def _build_hacc_command(self, nodes, nproc, particles, dir_):
        """构建HACC命令"""
        hacc_path = os.path.join(self.root_path, "io_apps/hacc_io_write")

        if self.fs_type == "GekkoFS":
            client_lib = os.path.join(self.config.get('gekkofs_path', 'gekkofs_home'),
                                      self.config.get('gekkofs_path', 'gekkofs_client'))
            return (f"mpirun -np {nproc} -hosts {nodes} "
                    f"-env LD_PRELOAD={client_lib}:$LD_PRELOAD "
                    f"{hacc_path} {particles} "
                    f"{os.path.join(dir_, 'testFile')}")
        else:
            return (f"mpirun -np {nproc} -hosts {nodes} "
                    f"{hacc_path} {particles} "
                    f"{os.path.join(dir_, 'testFile')}")
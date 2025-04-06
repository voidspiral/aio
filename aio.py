import configparser
import os
import time

from tuning.utils.get57features import *

from tuning.ior import searchBest
from ClusterShell.NodeSet import NodeSet

from tuning.utils.utils_else import *


def expand_nodes(node_list):
    # 使用 NodeSet 扩展节点列表，转化为mpich可使用的-hosts 参数
    nodeset = NodeSet(node_list)
    expanded_nodes = list(nodeset)
    return ','.join(str(node) for node in expanded_nodes)

def fold_nodes(node_list):
    nodeset = NodeSet(node_list)
    return str(nodeset)

def count_nodes(nodes_list):
    nodeset = NodeSet(nodes_list)
    return len(list(nodeset))

def load_config():
    config = configparser.ConfigParser()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(root_dir, "config", "storage.ini")
    config.read(config_path)
    return config


class AIO:
    def __init__(self, cmd, nodes_list):
        self.cmd = cmd
        self.ori_nodes_list = nodes_list

        # self.nodes_list = fold_nodes(nodes_list)
        # self.log_file = replace_spaces_with_underscores(get_file_name(cmd, self.nodes_list)) + ".txt"
        # self.log_file_2 = replace_spaces_with_underscores(get_file_name(cmd,self.nodes_list)) + ".txtt"
        # # print(self.log_file_2)
        # #TODO 扩展性的问题fix 节点列表过多的时候超过文件名称限制 比如32个节点cn[0-32] cn0,...,cn32
        # self.cmd_generate = replace_spaces_with_underscores(get_file_name(cmd,self.nodes_list)) + ".darshan"
        #backup
        self.nodes_list = expand_nodes(nodes_list)
        self.log_file = replace_spaces_with_underscores(cmd) + ".txt"
        self.log_file_2 = replace_spaces_with_underscores(cmd) + ".txtt"
        self.cmd_generate = replace_spaces_with_underscores(cmd) + ".darshan"
        self.nodes_count = count_nodes(nodes_list)
        self.final_name = f"N-{self.nodes_count}_"+self.log_file_2
        self.darshan_log = get_cwd() + "/darshan_log/"
        self.darshan_parse_log = get_cwd() + "/darshan_parse_log/"
        Path(self.darshan_log).mkdir(parents=True, exist_ok=True)
        Path(self.darshan_parse_log).mkdir(parents=True, exist_ok=True)

        self.runtime_AIO = 0
        self.speedup = 0
        self.bw = 0
        self.bw_AIO = 0
        self.tmpfs_path="/dev/shm"
        self.config = load_config()
        self.output = os.path.join(os.getcwd(), "result.txt")

    def run(self):
        self.retrieval_and_collect()
        self.execute()
        self.result()

    def retrieval_and_collect(self):
        if not self.check():
            self.run_with_darshan()

        with open(self.darshan_parse_log + f"N-{self.nodes_count}_"+self.log_file) as infile:
            for line in infile:
                # if line == '# MPI-IO module data\n':
                #    break
                if 'agg_perf_by_slowest' in line:
                    line = line.split()
                    self.bw = float(line[2])
                    break



    def execute(self):
        gekkofs_config, gekkofs_bw = self.execute_gekkofs()
        lustre_config, lustre_bw = self.execute_lustre()
        print(f"\n GekkoFS 配置 {gekkofs_config} \n，预测带宽: {gekkofs_bw:.2f} MB/s \n")
        print(f"\n Lustre 配置 {lustre_config} \n，预测带宽: {lustre_bw:.2f} MB/s \n")
        best_config, best_fs_type = None, None
        if gekkofs_bw > lustre_bw:
            best_config = gekkofs_config
            best_bw = gekkofs_bw
            best_fs_type = "GekkoFS"
            self.bw_AIO = best_bw
        else:
            best_config = lustre_config
            best_bw = lustre_bw
            best_fs_type = "Lustre"
            self.bw_AIO = best_bw

        print(f"\n选择 {best_fs_type} 作为最佳文件系统，预测带宽: {best_bw:.2f} MB/s")

    def result(self):
        if self.bw_AIO != 0:
            print("bw: ", self.bw)
            print("bw_AIO: ", self.bw_AIO)
            self.speedup = self.bw_AIO / self.bw
            print("AIO提升的加速比为：", self.speedup)
            with open(self.output, 'a') as f:
                f.write("\n")
                f.write("bw: %s" % str(self.bw))
                f.write("\n")
                f.write("bw_AIO: %s" % str(self.bw_AIO))
                f.write("\n")
                f.write("speedup: %s" % str(self.speedup))
                f.write("\n\n\n")

    def check(self):
        return search_file(self.darshan_log, self.cmd_generate)

    def run_with_darshan(self):
        self.darshan_init()
        print(self.cmd)
        result = subprocess.run(self.cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error executing command: {self.cmd}")
            print(f"Error output: {result.stderr}")
            raise RuntimeError(f"Command execution failed with return code {result.returncode}")
        self.darshan_parser()
        self.darshan_terminate()

    def darshan_init(self):
        os.environ["LD_PRELOAD"] = self.config.get('darshan_path', 'darshan_runtime')
        os.environ["DARSHAN_LOGFILE"] = self.darshan_log + self.cmd_generate

    def darshan_parser(self):
        os.environ["PATH"] = self.config.get('darshan_path', 'darshan_utils') + ":" + os.environ["PATH"]
        os.environ["LD_LIBRARY_PATH"] = self.config.get('darshan_path', 'darshan_ld_library') + ":" + os.environ[
            "LD_LIBRARY_PATH"]

        log_path = self.darshan_parse_log + f"N-{self.nodes_count}_"+self.log_file
        # print("OutPut file", self.darshan_parse_log + f"N-{self.nodes_count}"+self.log_file)
        command = "darshan-parser --show-incomplete --base --perf " + self.darshan_log + self.cmd_generate + " > " + log_path
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error executing command: {command}")
            print(f"Error output: {result.stderr}")
            raise RuntimeError(f"Command execution failed with return code {result.returncode}")
        log_path =  self.darshan_parse_log + f"N-{self.nodes_count}_"+self.log_file_2
        command = "darshan-parser --total --perf --file " + self.darshan_log + self.cmd_generate + " > " + log_path
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error executing command: {command}")
            print(f"Error output: {result.stderr}")
            raise RuntimeError(f"Command execution failed with return code {result.returncode}")

    def darshan_terminate(self):
        del os.environ["LD_PRELOAD"]


    def execute_tmpfs(self):
        print("选用的加速层为tmpfs")
        print("需要root权限")
        raise RuntimeError("need root")
        dst = get_last_slash_prefix(get_common_prefix(self.result_layer))
        src = self.tmpfs_path
        commands = [
            "sudo mount --bind " + src + " " + dst,
            self.cmd,
            "sudo umount " + dst
        ]
        os.environ["LD_LIBRARY_PATH"] = '/vol8/lzy/darshan-3.4.4-mpich/darshan-prefix/lib/' + ":" + os.environ[
            "LD_LIBRARY_PATH"]
        os.environ["PATH"] = '/vol8/lzy/darshan-3.4.4-mpich/darshan-prefix/bin/' + ":" + os.environ["PATH"]
        os.environ["LD_PRELOAD"] = '/vol8/lzy/darshan-3.4.4-mpich/darshan-runtime/lib/.libs/libdarshan.so' + ":" + \
                                   os.environ["LD_PRELOAD"]
        os.environ["DARSHAN_LOGPATH"] = '/vol8/lzy/AIO-main_lzy/'
        os.environ["DARSHAN_LOGFILE"] = '/vol8/lzy/AIO-main_lzy/ior_aio.darshan'
        start_time = time.time()
        subprocess.run(' && '.join(commands), shell=True, capture_output=False, text=True)
        self.runtime_AIO = time.time() - start_time

        subprocess.run(
            '/vol8/lzy/darshan-3.4.4-mpich/darshan-util/darshan-parser --show-incomplete --base --perf /vol8/lzy/AIO-main_lzy/ior_aio.darshan >> /vol8/lzy/AIO-main_lzy/ior_aio.txt',
            shell=True, capture_output=False, text=True)
        with open('/vol8/lzy/AIO-main_lzy/ior_aio.txt') as infile:
            for line in infile:
                if 'agg_perf_by_slowest' in line:
                    line = line.split()
                    self.bw_AIO = float(line[2])
                    break

    def execute_lustre(self):
        print("execute lustre")
        best_configlist, self.bw_AIO = searchBest(self.cmd, 'Lustre', self.cmd, self.ori_nodes_list, self.final_name)
        print("_________________________________________________________________________")
        print("best config for lustre:")
        print(best_configlist)
        with open(self.output, 'a') as f:
            f.write(str(best_configlist))
        return best_configlist, self.bw_AIO

    def execute_gekkofs(self):
        print("execute gekkofs")
        best_configlist, self.bw_AIO = searchBest(self.cmd, 'GekkoFS', self.cmd, self.ori_nodes_list, self.final_name)
        print("_________________________________________________________________________")
        print("best config for gekkofs:")
        with open(self.output, 'a') as f:
            f.write(str(best_configlist))
        return best_configlist, self.bw_AIO

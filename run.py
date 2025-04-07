import configparser
import os
import sys
import subprocess
def get_host_list():
    """获取主机列表"""
    try:
        result = subprocess.run('srun hostname | nodeset -f | nodeset -e',
                              shell=True,
                              capture_output=True,
                              text=True)
        if result.returncode == 0:
            hosts = result.stdout.strip().replace(" ", ",")
            return hosts
    except Exception as e:
        print(f"Error getting host list: {e}")
        return None

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("config/storage.ini")
    os.environ["PATH"] = "/thfs3/home/wuhuijun/lmj/fix/AIO_OPRAEL-20250219/io_apps:" + os.environ["PATH"]
    os.environ["PATH"] = config.get('mpi_path', 'mpirun') + ":" + os.environ["PATH"]
    hosts = get_host_list()
    print (hosts)
    cmd = f'python main.py "mpirun -np 16 -hosts {hosts} mpiior -a mpiio -w -e -t 16m -b 128m -i 5"'
    print ("test command", cmd)
    subprocess.run(cmd, shell=True, capture_output=False, text=True)


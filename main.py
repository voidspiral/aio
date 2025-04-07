import sys
import os
from aio import *


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
    cmd = get_cmd(sys.argv)

    hosts = get_host_list()
    print("hosts:", hosts)
    aio = AIO(cmd, hosts)
    aio.run()



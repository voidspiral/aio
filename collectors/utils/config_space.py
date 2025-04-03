from skopt.sampler import Lhs
from skopt.space import Space
import numpy as np


class ConfigSpace:
    """配置空间管理类"""

    def __init__(self, max_samples=500):
        """初始化配置空间管理类

        参数:
            max_samples: 要生成的最大样本数量，默认为500
        """
        self.max_samples = max_samples
        self.configs = self.init_config_space()

    def init_config_space(self):
        """初始化参数配置空间并使用拉丁超立方采样"""
        # 参数空间定义 - 直接使用实际参数范围
        spaces = {
            'romio': {
                'cb_read': [(0, 2)],
                'cb_write': [(0, 2)],
                'ds_read': [(0, 2)],
                'ds_write': [(0, 2)],
                'cb_nodes': [(1, 32)],
                'cb_config_list': [(1, 32)]
            },
            'lustre': {
                'stripe_size': [(1, 128)],  # 直接采样1-128 MB
                'stripe_count': [(1, 64)]
            },
            'gkfs': {
                'chunksize': [(1, 5)],  # 在512k 1m 2m 4m 8m的hash桶中选择index
                'dirents_buff_size': [(8, 16)],  # 直接采样1-16 MB
                'daemon_io_xstreams': [(8, 16)],  # 直接采样4-32
                'daemon_handler_xstreams': [(4, 8)]  # 直接采样2-16
            }
        }

        # 使用LHS生成采样点
        lhs = Lhs(criterion="maximin", iterations=500)
        configs = {}

        # 创建一个统一的随机索引，确保参数之间的对应关系
        # np.random.seed(42)  # 设置随机种子确保可重现性
        random_indices = np.random.permutation(self.max_samples)

        for category, params in spaces.items():
            configs[category] = {}

            # 先生成所有参数的样本
            all_samples = {}
            for param_name, bounds in params.items():
                # 创建scikit-optimize空间
                space = Space(bounds)
                # 生成采样点
                samples = lhs.generate(space.dimensions, self.max_samples)
                samples = np.array(samples).flatten()
                all_samples[param_name] = samples

            # 使用相同的随机索引重新排列所有参数，保持对应关系
            for param_name, samples in all_samples.items():
                configs[category][param_name] = samples[random_indices].tolist()

        return configs

    def get_romio_config(self, count):
        """获取ROMIO配置"""
        return [
            self.configs['romio']['cb_read'][count],
            self.configs['romio']['cb_write'][count],
            self.configs['romio']['ds_read'][count],
            self.configs['romio']['ds_write'][count],
            self.configs['romio']['cb_nodes'][count],
            self.configs['romio']['cb_config_list'][count]
        ]

    def get_lustre_config(self, count):
        """获取Lustre配置"""
        return [
            self.configs['lustre']['stripe_size'][count],
            self.configs['lustre']['stripe_count'][count]
        ]

    def get_gkfs_config(self, count):
        """获取GekkoFS配置"""
        return [
            self.configs['gkfs']['chunksize'][count],
            self.configs['gkfs']['dirents_buff_size'][count],
            self.configs['gkfs']['daemon_io_xstreams'][count],
            self.configs['gkfs']['daemon_handler_xstreams'][count]
        ]


if __name__ == '__main__':
    # 测试LHS采样
    config_space = ConfigSpace(max_samples=5)
    print("Sample Lustre stripe sizes:", config_space.configs['lustre']['stripe_size'][:10])
    print("Sample GekkoFS chunk sizes:", config_space.configs['gkfs']['chunksize'][:10])

    # 验证所有参数
    print("\n验证参数范围:")
    for category in ['romio', 'lustre', 'gkfs']:
        for param, values in config_space.configs[category].items():
            print(type(values[0]))
            print(f"{category}.{param} 范围: [{min(values)}, {max(values)}], 前5个值: {values[:5]}")
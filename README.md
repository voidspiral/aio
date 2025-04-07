# AIO (Automatic I/O Optimization)

自动I/O优化系统，用于并行文件系统性能优化。支持Lustre和GekkoFS文件系统，通过智能参数调优提升I/O性能。

## 离线安装
离线提供的第三方安装包
python3.8
```
export PYTHONHOME=/yourpath/python3.8
export PYTHONPATH=/yourpath/lib/python3.8:/yourpath/python3.8/dist-packages:/yourpath/local/lib/python3.8/dist-packages
cp /usr/bin/usr-bin-python3.8
ln -s /usr/bin/usr-bin-python3.8 python3.8
//测试是否导入成功
python3.8 进入交互式shell 
import numpy
import pandas
import sys
//显示路径即可
print(sys.path)
```
### Gekkofs 部署
git clone -b th-compile-passed https://github.com/sktzwhj/gekkofs-v0.9.2.git
### Lustre 部署
略
## 功能特性

- 支持多种并行文件系统（Lustre、GekkoFS）
- 自动参数调优
- 基准测试数据收集（IOR、HACC-IO）
- MPI-IO优化 romio

## 系统要求

- Python 3.8+
- MPI环境（MPICH4推荐） 使用前先加载环境变量
- Darshan工具链
- 并行文件系统（Lustre/GekkoFS）

## 目录结构
```
aio/                           # 项目根目录
├── collector.py                # 主收集器脚本
├── run.py                      # 运行脚本
├── main.py                     # 主入口脚本
├── aio.py                      # AIO 核心功能实现
├── requirements.txt            # 项目依赖
├── README.md                   # 项目说明文件
├── result.txt                  # 结果文件
├── collectors/                 # 收集器模块目录
│   ├── __init__.py             # 收集器模块初始化文件
│   ├── feature_processor.py    # 特征处理器
│   ├── base_collector.py       # 基础收集器类
│   ├── ior_collector.py        # IOR 收集器
│   ├── hacc_collector.py       # HACC 收集器
│   ├── lustre.csv              # Lustre 特征数据
│   ├── gekkofs.csv             # GekkoFS 特征数据
│   └── utils/                  # 收集器工具目录
│       ├── __init__.py         # 工具初始化文件
│       ├── get57features.py    # 57 特征提取器
│       ├── config_space.py     # 配置空间定义
│       └── sampling.py         # 采样工具
├── config/                     # 配置文件目录
│   └── storage.ini             # 存储配置文件
├── darshan_log/                # Darshan 日志目录
│   └── *.darshan               # Darshan 日志文件
├── darshan_parse_log/          # 解析后的 Darshan 日志目录
│   └── *.txt                   # 解析后的日志文件
├── io_apps/                    # I/O 应用程序目录
│   ├── mpiior                  # MPI IOR 应用程序
│   ├── hacc_io                 # HACC I/O 应用程序
│   └── hacc_io_write           # HACC I/O 写入应用程序
├── logs/                       # 日志目录
├── model_training/             # 模型训练目录
│   ├── train_model.py          # 模型训练脚本
│   ├── feature.py              # 特征定义
│   └── models/                 # 训练好的模型目录
│       ├── slModel.pkl         # Lustre 模型
│       └── sgModel.pkl         # GekkoFS 模型
├── oprael/                     # OPRAEL 优化器目录
│   ├── optimizer/              # 优化器子目录
│   │   ├── ior_optimizer.py    # IOR 优化器
└── tuning/                     # 调优目录
    ├── __init__.py             # 调优初始化文件
    ├── ior.py                  # IOR 调优脚本
    ├── hint.txt                # 提示文件
    ├── romio/                  # ROMIO 调优目录
    └── utils/                  # 调优工具目录
```
# tuning/mpiio.c

```shell 
module load mpich4
cd tuning/romio
make
```


# Data Collection
salloc 分配最大运行节点
运行IOR基准测试
./collector.py -t Lustre -w "node[1-4]" -o ./dataset_lustre -b ior

./collector.py -t Lustre  -o ./collectors/logs/lustre -b ior -w  `srun hostname | nodeset -f`

./collector.py -t GekkoFS  -o ./collectors/logs/gekkofs -b ior -w  `srun hostname | nodeset -f`



# Models For Autotuning File Systems

python model_training/train_model.py


# Running

python run.py



### *Note*


## 参数说明

### Lustre参数
- stripe_size: 条带大小（MB）
- stripe_count: 条带数量

### GekkoFS参数
- chunk_size: 块大小（KB）
- dirents_buff_size: 目录缓冲区大小（MB）
- daemon_io_xstreams: IO线程数
- daemon_handler_xstreams: 处理线程数

### ROMIO参数
- cb_nodes: 聚合节点数
- romio_cb_read: 读取模式
- romio_cb_write: 写入模式
- romio_ds_read: 读取数据传输模式
- romio_ds_write: 写入数据传输模式

## 注意事项

1. 运行环境
- 确保MPI环境正确配置
- 检查文件系统挂载状态
- 预留足够存储空间


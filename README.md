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

# tuning/mpiio.c
修改romio参数，注意将参数路径
先编译
```shell

```

# Data Collection

salloc 分配最大运行节点
运行IOR基准测试
./collector.py -t Lustre -w "node[1-4]" -o ./dataset_lustre -b ior

./collector.py -t Lustre  -o ./dataset_lustre -b ior -w  `srun hostname | nodeset -f`

./collector.py -t GekkoFS  -o ./dataset_lustre -b ior -w  `srun hostname | nodeset -f`


# Model For File System Selection

python model_trainning/train_mode.py



# Models For Autotuning File Systems

python AIO_searcher-model.py ./dataset_lustre Lustre 1.csv

python AIO_searcher-model.py ./dataset_gekkfs GekkoFS 1.csv



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


3. 故障排除
- 检查日志文件
- 验证配置参数
- 确认权限设置

## 开发说明

1. 添加新的测试模块
- 参考现有模块结构
- 实现数据收集接口
- 更新配置文件

2. 参数调优
- 在tuning目录下修改参数
- 遵循配置文件格式
- 测试参数有效性


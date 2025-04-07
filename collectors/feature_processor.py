import sys
import logging
import pandas as pd
import os
import subprocess
import glob
import shutil
import configparser

from utils.get57features import extracting_darshan

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config(config_path: str = None) -> configparser.ConfigParser:
    """加载配置文件"""
    config = configparser.ConfigParser()
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        root_dir = os.path.dirname(current_dir)

        config_path = os.path.join(root_dir, "config", "storage.ini")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config.read(config_path)
    return config


def setup_environment(config: configparser.ConfigParser):
    """设置环境变量"""
    try:
        # 设置PATH环境变量
        darshan_utils_path = config.get('darshan_path', 'darshan_utils')
        os.environ["PATH"] = darshan_utils_path + ":" + os.environ["PATH"]

        # 设置LD_LIBRARY_PATH环境变量
        darshan_lib_path = config.get('darshan_path', 'darshan_ld_library')
        os.environ["LD_LIBRARY_PATH"] = darshan_lib_path + ":" + os.environ["LD_LIBRARY_PATH"]

        logging.info("Environment variables set successfully")
    except Exception as e:
        logging.error(f"Error setting environment variables: {e}")
        raise


def parse_darshan_directory(input_dir: str, temp_dir: str) -> None:
    """解析目录下所有 Darshan 文件"""
    os.makedirs(temp_dir, exist_ok=True)

    for file in glob.glob(os.path.join(input_dir, "*")):
        if os.path.isfile(file):
            try:
                output_file = os.path.join(temp_dir, f"{os.path.basename(file)}.txt")
                command = f"darshan-parser --total --file --perf {file}"

                with open(output_file, "w") as outfile:
                    result = subprocess.run(command, shell=True, stdout=outfile, stderr=subprocess.PIPE, text=True)

                if result.returncode == 0:
                    logging.info(f"Successfully parsed {file}")
                else:
                    logging.error(f"Failed to parse {file}: {result.stderr}")

            except Exception as e:
                logging.error(f"Error processing {file}: {e}")


def extract_features(temp_dir: str) -> pd.DataFrame:
    """从解析数据中提取特征"""
    try:
        # 使用 feature_processor 的方法提取特征
        df = extracting_darshan(os.path.join(temp_dir, "*.txt"))
        return df
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise


def main():
    """主函数：处理 Lustre 和 GekkoFS 数据"""
    try:
        # 加载配置并设置环境
        config = load_config()
        setup_environment(config)

        # 处理 Lustre 数据
        lustre_temp = "temp_lustre"
        logging.info("Processing Lustre dataset...")
        parse_darshan_directory("./logs/lustre", lustre_temp)
        df_lustre = extract_features(lustre_temp)
        df_lustre.to_csv("lustre.csv", index=False)
        logging.info("Lustre features saved to lustre.csv")

        # 清理 Lustre 临时文件
        shutil.rmtree(lustre_temp)

        # 处理 GekkoFS 数据
        gekkofs_temp = "temp_gekkofs"
        logging.info("Processing GekkoFS dataset...")
        parse_darshan_directory("./logs/gekkofs", gekkofs_temp)
        df_gekkofs = extract_features(gekkofs_temp)
        df_gekkofs.to_csv("gekkofs.csv", index=False)
        logging.info("GekkoFS features saved to gekkofs.csv")

        # 清理 GekkoFS 临时文件
        shutil.rmtree(gekkofs_temp)

        logging.info("All processing completed successfully")

    except Exception as e:
        logging.error(f"Error in main process: {e}")
        # 确保清理临时目录
        for temp_dir in [lustre_temp, gekkofs_temp]:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
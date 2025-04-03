#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from collectors import HACCCollector, IORCollector


def normalize_path(path):
    return Path(path).resolve()

def main():
    parser = argparse.ArgumentParser(description='AIO Data Collection')
    parser.add_argument('-t', '--type',
                        type=str,
                        choices=['Lustre', 'GekkoFS', 'else'],
                        required=True,
                        help='The storage layer type')
    parser.add_argument('-w', '--nodelist',
                        type=str,
                        required=True,
                        help="Nodelist for running tests")
    parser.add_argument('-o', '--output',
                        type=str,
                        required=True,
                        help='The output path of darshan')
    parser.add_argument('-b', '--benchmark',
                        type=str,
                        choices=['ior', 'hacc'],
                        required=True,
                        help='Benchmark type to run')

    args = parser.parse_args()

    # 选择收集器
    if args.benchmark == 'ior':
        collector = IORCollector(
            fs_type=args.type,
            nodelist=args.nodelist,
            output=normalize_path(args.output)
        )
    elif args.benchmark == 'hacc':
        collector = HACCCollector(
            fs_type=args.type,
            nodelist=args.nodelist,
            output=normalize_path(args.output)
        )
    else:
        pass
    # 运行收集
    collector.collect_trace()


if __name__ == "__main__":
    main()
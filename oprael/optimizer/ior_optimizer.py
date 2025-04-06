# License: MIT

import time
import math
from typing import List
from tqdm import tqdm
import numpy as np
from oprael import logger
from oprael.optimizer.base import BOBase
from oprael.utils.constants import SUCCESS, FAILED, TIMEOUT
from oprael.utils.limit import TimeoutException
from oprael.utils.util_funcs import deprecate_kwarg
from oprael.utils.history import Observation, History
import os
import pandas as pd
import configparser
from concurrent.futures import ThreadPoolExecutor
from oprael.utils.aio import log_features, perc_features, romio_features, lustre_feature, gkfs_feature
import pickle

from tuning.utils.get57features import extracting_darshan57


class Ensemble(BOBase):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            objective_function: callable,
            cmd,
            fs_type,
            config_space_lustre,
            config_space_gekkofs,
            ori_nodelist,
            log_file_2,
            best_config,
            best_perf,
            model,
            config_space,
            num_objectives=1,
            num_constraints=0,
            sample_strategy: str = 'bo',
            max_runs=200,
            runtime_limit=None,
            time_limit_per_trial=180,
            advisor_type='default',
            surrogate_type='auto',
            acq_type='auto',
            iteration_id=0,
            acq_optimizer_type='auto',
            initial_runs=3,
            init_strategy='random_explore_first',
            initial_configurations=None,
            ref_point=None,
            transfer_learning_history: List[History] = None,
            logging_dir='logs',
            task_id='OPRAEL',
            visualization='none',
            auto_open_html=False,
            random_state=None,
            logger_kwargs: dict = None,
            advisor_kwargs: dict = None,
            train_csv: str = None,
            access: str = "w",
            darshan_path: str = None,
            common_feature: dict = None,
            custom_advisor_list: list = None,
    ):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.FAILED_PERF = [np.inf] * num_objectives
        # super().__init__(objective_function, cmd, config_space, best_config, best_perf, model_pkl, model, task_id=task_id, output_dir=logging_dir,
        # super().__init__(objective_function, cmd, config_space, best_config, best_perf, task_id=task_id, output_dir=logging_dir,
        #                 random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
        #                 runtime_limit=runtime_limit, sample_strategy=sample_strategy,
        #                 time_limit_per_trial=time_limit_per_trial, transfer_learning_history=transfer_learning_history,
        #                 logger_kwargs=logger_kwargs)
        super().__init__(objective_function, config_space)

        self.best_config = None
        self.best_perf = 0
        self.common_feature = common_feature
        self.advisor_type = advisor_type
        self.Mbytes = 1024 * 1024
        advisor_kwargs = advisor_kwargs or {}
        _logger_kwargs = {'force_init': False}  # do not init logger in advisor
        self.custom_advisor_list = custom_advisor_list
        self.start = time.time()
        self.iteration_id = 0
        self.max_runs = max_runs
        self.cmd = cmd
        self.fs_type = fs_type
        self.global_config = self.load_config()
        #aio中的log_file_2
        self.log_file_2 = log_file_2

        root_path = self.global_config.get("root_path", "home")
        if self.fs_type == 'Lustre':
            self.config_space = config_space_lustre
            with open(f"{root_path}/model_training/models/slModel.pkl", 'rb') as f:
                self.model = pickle.load(f)
        elif self.fs_type == 'GekkoFS':
            self.config_space = config_space_gekkofs
            with open(f"{root_path}/model_training/models/sgModel.pkl", 'rb') as f:
                self.model = pickle.load(f)

        if advisor_type == 'default':
            from oprael.core.tpe_advisor import TPE_Advisor
            from oprael.core.ga_advisor import GA_Advisor
            self.config_advisor_list = []
            self.pool = ThreadPoolExecutor(max_workers=2)
            ga = GA_Advisor(config_space,
                            num_objectives=num_objectives,
                            num_constraints=num_constraints,
                            optimization_strategy=sample_strategy,
                            batch_size=1,
                            task_id=task_id,
                            output_dir=logging_dir,
                            random_state=random_state,
                            logger_kwargs=_logger_kwargs,
                            **advisor_kwargs)

            tpe = TPE_Advisor(config_space, task_id=task_id, random_state=random_state,
                              logger_kwargs=_logger_kwargs, **advisor_kwargs)
            self.config_advisor_list.append(tpe)
            self.config_advisor_list.append(ga)
        elif advisor_type == 'custom':  # Customizable additions and deletions
            self.pool = ThreadPoolExecutor(max_workers=len(self.custom_advisor_list))
            self.config_advisor_list = []

            for _ in self.custom_advisor_list:
                if _ == "bo":
                    from oprael.core.bo_advisor import Advisor
                    self.config_advisor_list.append(Advisor(config_space,
                                                            num_objectives=num_objectives,
                                                            num_constraints=num_constraints,
                                                            initial_trials=initial_runs,
                                                            init_strategy=init_strategy,
                                                            initial_configurations=initial_configurations,
                                                            optimization_strategy=sample_strategy,
                                                            surrogate_type=surrogate_type,
                                                            acq_type=acq_type,
                                                            acq_optimizer_type=acq_optimizer_type,
                                                            ref_point=ref_point,
                                                            transfer_learning_history=transfer_learning_history,
                                                            task_id=task_id,
                                                            output_dir=logging_dir,
                                                            random_state=random_state,
                                                            logger_kwargs=_logger_kwargs,
                                                            **advisor_kwargs))

                elif _ == "tpe":
                    from oprael.core.tpe_advisor import TPE_Advisor
                    self.config_advisor_list.append(
                        TPE_Advisor(config_space, task_id=task_id, random_state=random_state,
                                    logger_kwargs=_logger_kwargs, **advisor_kwargs))

                elif _ == "ga":
                    from oprael.core.ga_advisor import GA_Advisor
                    self.config_advisor_list.append(GA_Advisor(config_space,
                                                               num_objectives=num_objectives,
                                                               num_constraints=num_constraints,
                                                               optimization_strategy=sample_strategy,
                                                               batch_size=1,
                                                               task_id=task_id,
                                                               output_dir=logging_dir,
                                                               random_state=random_state,
                                                               logger_kwargs=_logger_kwargs,
                                                               **advisor_kwargs))
        else:
            raise ValueError('Invalid advisor type!')

    def run(self) -> History:
        for _ in tqdm(range(self.iteration_id, self.max_runs)):
            if self.budget_left < 0:
                logger.info('Time %f elapsed!' % self.runtime_limit)
                break
            start_time = time.time()
            self.iterate(budget_left=self.budget_left)
            runtime = time.time() - start_time
            self.budget_left -= runtime
        return self.get_history(), self.best_config, self.best_perf

    @staticmethod
    def load_config():
        config = configparser.ConfigParser()
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(root_dir, "config", "storage.ini")
        config.read(config_path)
        return config

    def parallel_get_suggestion(self, advisor):
        """
        并行获取建议配置
        """
        config = advisor.get_suggestion()

        darshan_parse_log = os.path.join(os.getcwd(),"darshan_parse_log")
        # print("!!!", os.path.join(darshan_parse_log, self.log_file_2))
        if self.fs_type == 'Lustre':
            dfconfig = pd.DataFrame([config], columns=romio_features + lustre_feature)
            for c in dfconfig:
                dfconfig[c] = np.log10(dfconfig[c] + 1.1).fillna(value=-10)

            dfs = extracting_darshan57(os.path.join(darshan_parse_log, self.log_file_2))  # 引入utils/get57features.py

            # dfs = FeatureProcessor.extracting_darshan57(os.path.join(darshan_parse_log, log_file_2))
            # print("\n=== 检查dfs的列名 ===")
            # print("dfs columns:", dfs.columns.tolist())
            # print("NODES in dfs:", 'NODES' in dfs.columns)

            df = pd.concat([dfs, dfconfig], axis=1)
            # print("\n=== 检查合并后df的列名 ===")
            # print("df columns:", df.columns.tolist())
            # print("NODES in df:", 'NODES' in df.columns)

            # print("\n=== 检查特征列表 ===")
            all_features = log_features + perc_features + romio_features + lustre_feature
            # print("all_features:", all_features)
            # print("NODES in all_features:", 'NODES' in all_features)

            X = df[log_features + perc_features + romio_features + lustre_feature]
            # print("\n=== 检查最终X的列名 ===")
            # print("X columns:", X.columns.tolist())
            # print("NODES in X:", 'NODES' in X.columns)

            performance = self.model.predict(X)
            return [config, performance]
        elif self.fs_type == 'GekkoFS':
            dfconfig = pd.DataFrame([config], columns=romio_features + gkfs_feature)
            for c in dfconfig:
                dfconfig[c] = np.log10(dfconfig[c] + 1.1).fillna(value=-10)


            dfs = extracting_darshan57(os.path.join(darshan_parse_log, self.log_file_2))  # 引入utils/get57features.py

            # dfs = FeatureProcessor.extracting_darshan57(os.path.join(darshan_parse_log, log_file_2))
            # print("\n=== 检查dfs的列名 ===")
            # print("dfs columns:", dfs.columns.tolist())
            # print("NODES in dfs:", 'NODES' in dfs.columns)

            df = pd.concat([dfs, dfconfig], axis=1)
            # print("\n=== 检查合并后df的列名 ===")
            # print("df columns:", df.columns.tolist())
            # print("NODES in df:", 'NODES' in df.columns)
            #
            # print("\n=== 检查特征列表 ===")
            all_features = log_features + perc_features + romio_features + gkfs_feature
            # print("all_features:", all_features)
            # print("NODES in all_features:", 'NODES' in all_features)

            X = df[log_features + perc_features + romio_features + gkfs_feature]
            # print("\n=== 检查最终X的列名 ===")
            # print("X columns:", X.columns.tolist())
            # print("NODES in X:", 'NODES' in X.columns)

            performance = self.model.predict(X)
            return [config, performance]
        else:
            pass

    def score(self):
        tasks = []
        for adv_ in self.config_advisor_list:
            tasks.append(self.pool.submit(self.parallel_get_suggestion, adv_))

        results = []

        for task_ in tasks:
            t_result = task_.result()
            results.append(t_result)

        # Take maximum
        max_score = 0
        max_id = 0

        for i in range(len(results)):
            performance = results[i][1]
            if performance > max_score:  # or max_score * 1.2: consider a 20% error
                max_score = performance
                max_id = i

        return results[max_id][0], -results[max_id][1]

    def iterate(self, budget_left=None) -> Observation:
        # get configuration suggestion from advisor

        config, objectives_real = self.score()

        trial_state = SUCCESS
        _budget_left = int(1e10) if budget_left is None else budget_left
        _time_limit_per_trial = math.ceil(min(self.time_limit_per_trial, _budget_left))

        cost_time = time.time() - self.start
        logger.info("cost_time:%f" % (cost_time))
        start_time = time.time()
        try:
            # evaluate configuration on objective_function within time_limit_per_trial
            args, kwargs = (config, self.fs_type, self.cmd), dict()
            # timeout_status, _result = time_limit(self.objective_function,
            #                                      _time_limit_per_trial,
            #                                      args=args, kwargs=kwargs)
            # 0 config 1 performance
            # 不搜索参数直接推理
            objectives = self.score()[1]
            # if timeout_status:
            #     raise TimeoutException(
            #         'Timeout: time limit for this evaluation is %.1fs' % _time_limit_per_trial)
            # else:
            #     # parse result
            # objectives, constraints, extra_info = parse_result(_result)
            trial_state = 1
        except Exception as e:
            # parse result of failed trial
            if isinstance(e, TimeoutException):
                logger.warning(str(e))
                trial_state = TIMEOUT
            else:  # todo: log exception if objective function raises error
                logger.warning(f'Exception when calling objective function: {e}\nconfig: {config}')
                trial_state = FAILED
            objectives = self.FAILED_PERF
            constraints = None
            extra_info = None

        if objectives[0] < self.best_perf:
            self.best_config = config
            self.best_perf = objectives[0]

        elapsed_time = time.time() - start_time
        # update observation to advisor
        observation = Observation(
            config=config, objectives=objectives,
            trial_state=trial_state, elapsed_time=elapsed_time,
        )
        if _time_limit_per_trial != self.time_limit_per_trial and trial_state == TIMEOUT:
            # Timeout in the last iteration.
            pass
        else:
            for adv_ in self.config_advisor_list:
                adv_.update_observation(observation)

        self.iteration_id += 1
        # Logging
        if self.num_constraints > 0:
            logger.info('Iter %d, objectives: %s. constraints: %s.'
                        % (self.iteration_id, objectives, constraints))
        else:
            logger.info('Iter %d, objectives: %s.' % (self.iteration_id, objectives))

        return observation

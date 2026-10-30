import random
from typing import List, cast
from os.path import join
from argparse import ArgumentParser
import os
import json
from src.utils import unique_xfg_sym, split_list
import networkx as nx
from src.preprocess.symbolizer import clean_gadget
from tqdm import tqdm
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf, DictConfig
from multiprocessing import cpu_count, Manager, Pool, Queue
import functools
import dataclasses
from src.preprocess.symbolizer import tokenize_code_line

USE_CPU = cpu_count()


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/vd.yaml",
                            type=str)
    return arg_parser


def unique_data(cweid: str, root: str):
    """
    unique raw data without symbolization
    Args:
        cweid:dataset name
        root:data folder

    Returns:

    """
    XFG_root_path = join(root, cweid, "XFG")
    testcaseids = os.listdir(XFG_root_path)
    xfg_paths = list()
    for testcase in testcaseids:
        testcase_root_path = join(XFG_root_path, testcase)
        k_root_path = join(testcase_root_path, "suspicious")
        xfg_ps = os.listdir(k_root_path)
        for xfg_p in xfg_ps:
            xfg_path = join(k_root_path, xfg_p)
            xfg_paths.append(xfg_path)
    # remove duplicates and conflicts
    xfg_dict = unique_xfg_sym(xfg_paths)
    xfg_unique_paths = list()
    for md5 in xfg_dict:
        # remove conflicts
        if xfg_dict[md5]["label"] != -1:
            xfg_unique_paths.append(xfg_dict[md5]["xfg"])
    return xfg_unique_paths


@dataclasses.dataclass
class QueueMessage:
    XFG: nx.DiGraph
    xfg_path: str
    to_remove: bool = False
    is_finished: bool = False


def handle_queue_message(queue: Queue):
    """

    Args:
        queue:

    Returns:

    """
    xfg_ct = 0
    while True:
        message: QueueMessage = queue.get()
        if message.is_finished:
            break
        if message.to_remove:
            os.system(f"rm {message.xfg_path}")
        else:
            if message.XFG is not None:
                nx.write_gpickle(message.XFG, message.xfg_path)
                xfg_ct += 1
    return xfg_ct


def get_indentation(file_contents):
    indentation = []
    for statement in file_contents:
        count = 0
        for t in statement:
            if t == '\t':
                count += 1
            else:
                break
        indentation.append(count)
    return indentation


def get_condition(file_contents, indentations):
    condition = []
    flag_list = []
    for i in range(len(indentations)):  # init
        condition.append('N')
        flag_list.append(0)
    current_indentation = 0
    while 'N' in condition:
        idx = 0
        area_flag = 0  # 0->none; 1->'if' and 'else if'; 2->'else'
        pred_idx = 0
        while idx < len(indentations):
            if area_flag == 0:
                if indentations[idx] == current_indentation:
                    if condition[idx] == 'N':
                        condition[idx] = 'U'
                    if ("if" in file_contents[idx]) and ("else" not in file_contents[idx]):
                        area_flag = 1
                        pred_idx = idx
                    if ("if" in file_contents[idx]) and ("else" in file_contents[idx]):
                        area_flag = 1
                        pred_idx = idx
                    if ("if" not in file_contents[idx]) and ("else" in file_contents[idx]):
                        area_flag = 2
                        pred_idx = idx
            elif area_flag == 1:  # True area
                if indentations[idx] == indentations[idx - 1] == current_indentation and '{' in file_contents[idx]:  # the case of '{' after if
                    if condition[idx] == 'N':
                        condition[idx] = 'U'
                if indentations[idx] == current_indentation + 1:
                    if condition[idx] == 'N':
                        condition[idx] = 'T'
                if indentations[idx] == current_indentation and idx - pred_idx > 1:
                    if condition[idx] == 'N':
                        condition[idx] = 'U'
                    if ("if" in file_contents[idx]) and ("else" not in file_contents[idx]):
                        area_flag = 1
                        pred_idx = idx
                    elif ("if" in file_contents[idx]) and ("else" in file_contents[idx]):
                        area_flag = 1
                        pred_idx = idx
                    elif ("if" not in file_contents[idx]) and ("else" in file_contents[idx]):
                        area_flag = 2
                        pred_idx = idx
                    else:
                        area_flag = 0
                        pred_idx = 0
            else:  # area_flag == 2:  # False area
                if indentations[idx] == indentations[idx - 1] == current_indentation and '{' in file_contents[idx]:  # the case of '{' after if
                    if condition[idx] == 'N':
                        condition[idx] = 'U'
                if indentations[idx] == current_indentation + 1:
                    if condition[idx] == 'N':
                        condition[idx] = 'F'
                if indentations[idx] == current_indentation and idx - pred_idx > 1:
                    if condition[idx] == 'N':
                        condition[idx] = 'U'
                    if ("if" in file_contents[idx]) and ("else" not in file_contents[idx]):
                        area_flag = 1
                        pred_idx = idx
                    elif ("if" in file_contents[idx]) and ("else" in file_contents[idx]):
                        area_flag = 1
                        pred_idx = idx
                    elif ("if" not in file_contents[idx]) and ("else" in file_contents[idx]):
                        area_flag = 2
                        pred_idx = idx
                    else:
                        area_flag = 0
                        pred_idx = 0
            idx += 1
        current_indentation += 1
    return condition


def process_parallel(testcaseid: str, queue: Queue, XFG_root_path: str, split_token: bool):
    """

    Args:
        testcaseid:
        queue:
        XFG_root_path:

    Returns:

    """
    testcase_root_path = join(XFG_root_path, testcaseid)
    k_root_path = join(testcase_root_path, "suspicious")
    xfg_ps = os.listdir(k_root_path)
    for xfg_p in xfg_ps:
        xfg_path = join(k_root_path, xfg_p)
        xfg: nx.DiGraph = nx.read_gpickle(xfg_path)
        file_path = xfg.graph["file_paths"][0]
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_contents = f.readlines()
        indentation = get_indentation(file_contents)
        condition = get_condition(file_contents, indentation)
        code_lines = list()
        for n in xfg:
            code_lines.append(file_contents[n - 1])

        sym_code_lines = clean_gadget(code_lines)

        to_remove = list()
        for idx, n in enumerate(xfg):
            xfg.nodes[n]["code_sym_token"] = tokenize_code_line(sym_code_lines[idx], split_token)
            xfg.nodes[n]["indentation"] = indentation[n-1]
            xfg.nodes[n]["condition"] = condition[n-1]
            if len(xfg.nodes[n]["code_sym_token"]) == 0:
                to_remove.append(n)
        xfg.remove_nodes_from(to_remove)

        if len(xfg.nodes) != 0:
            queue.put(QueueMessage(xfg, xfg_path))
        else:
            queue.put(QueueMessage(xfg, xfg_path, to_remove=True))
    return testcaseid


def add_symlines(cweid: str, root: str, split_token: bool):
    """

    Args:
        cweid:
        root:

    Returns:

    """

    XFG_root_path = join(root, cweid, "XFG")
    testcaseids = os.listdir(XFG_root_path)
    testcase_len = len(testcaseids)

    with Manager() as m:
        message_queue = m.Queue()  # type: ignore
        pool = Pool(USE_CPU)
        xfg_ct = pool.apply_async(handle_queue_message, (message_queue,))
        process_func = functools.partial(process_parallel, queue=message_queue, XFG_root_path=XFG_root_path,
                                         split_token=split_token)
        testcaseids_done: List = [
            testcaseid
            for testcaseid in tqdm(
                pool.imap_unordered(process_func, testcaseids),
                desc=f"testcases: ",
                total=testcase_len,
            )
        ]

        message_queue.put(QueueMessage(None, None, False, True))
        pool.close()
        pool.join()
    print(f"total {xfg_ct.get()} XFGs!")


if __name__ == '__main__':
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    config = cast(DictConfig, OmegaConf.load(__args.config))
    seed_everything(config.seed, workers=True)
    data_folder = config.data_folder
    dataset_name = config.dataset.name
    root = os.path.join(data_folder, dataset_name)
    with open(os.path.join(data_folder, dataset_name, "split_info.json"), 'r') as f:
        split_info = json.load(f)
    train_cases_list = split_info["train"]
    val_cases_list = split_info["val"]
    test_cases_list = split_info["test"]

    xfg_train_list = []
    xfg_val_list = []
    xfg_test_list = []

    print(dataset_name)
    add_symlines(dataset_name, data_folder, config.split_token)
    print("begin unique_data")
    xfg_unique_paths = unique_data(dataset_name, data_folder)
    print("begin split_dataset")

    for xfg_path in tqdm(xfg_unique_paths, total=len(xfg_unique_paths), desc=f"pdgs"):
        xfg_dir_list = xfg_path.split(os.sep)
        xfg_caseid = xfg_dir_list[-3]
        if xfg_caseid in train_cases_list:
            xfg_train_list.append(xfg_path)
        if xfg_caseid in val_cases_list:
            xfg_val_list.append(xfg_path)
        if xfg_caseid in test_cases_list:
            xfg_test_list.append(xfg_path)
    with open(f"{root}/train.json", "w") as f:
        json.dump(xfg_train_list, f)
        print("train lens:%d" % len(xfg_train_list))
    with open(f"{root}/val.json", "w") as f:
        json.dump(xfg_val_list, f)
        print("val len:%d" % len(xfg_val_list))
    with open(f"{root}/test.json", "w") as f:
        json.dump(xfg_test_list, f)
        print("test lens:%d" % len(xfg_test_list))



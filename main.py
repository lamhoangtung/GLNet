import os
import argparse
import random
import datetime
import subprocess
import numpy as np
import time

import torch
import torch.backends.cudnn as cudnn

import utils.config_loader as loader
from utils.utils import unzip
from utils.utils import write_log
import processes.workflow as workflow

parser = argparse.ArgumentParser(description='OCR')
parser.add_argument('--config')


def unzip_data(configs):
    if configs["data_zip_file"]:
        zip_file = configs["data_zip_file"]
        folder = os.path.dirname(zip_file)
        print("Doing unzip file {0}:".format(zip_file))
        unzip(zip_file, folder)


def main():
    if os.path.isfile('requirements.txt'):
        subprocess.run('pip install -r requirements.txt', shell=True)

    args = parser.parse_args()
    configs = loader.load_config(args.config)

    # Seed and GPU setting
    random.seed(configs["manual_seed"])
    np.random.seed(configs["manual_seed"])
    torch.manual_seed(configs["manual_seed"])

    if torch.cuda.is_available():
        if configs["gpu"] is not None:
            device = torch.device("cuda:" + str(configs["gpu"]))
            configs["device"] = device
            torch.cuda.manual_seed(configs["manual_seed"])
            cudnn.benchmark = True
            cudnn.deterministic = True
        else:
            device = torch.device("cpu")
            configs["device"] = device
    else:
        device = torch.device("cpu")
        configs["device"] = device

    # set name experiment
    if not configs["experiment_name"]:
        train_file_name = configs["train_data_file"].split("/")[-1].split(".")[0]
        configs["experiment_name"] = "{0}-{1}-ctc-{2}".\
            format(configs["feature_extraction"], configs["sequence_modeling"], train_file_name)

    if configs["saved_models"] is None or configs["saved_models"] == "":
        time_now = datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        saved_models = os.path.join("./saved_models", configs["experiment_name"] + time_now)
        saved_models = os.path.abspath(saved_models)
        configs["saved_models"] = saved_models
    os.makedirs(configs["saved_models"], exist_ok=True)

    # view folders
    if os.path.exists("/opt/ml/input/data/train"):
        files_in = os.listdir("/opt/ml/input/data/train")
        if len(files_in) > 0:
            content = "/opt/ml/input/data/train:\n"
            for f in files_in:
                content += f + "\n"
            write_log(os.path.join(configs["saved_models"], "opt.txt"), content)
            print(content)
    if os.path.exists("/opt/ml/code"):
        files_in = os.listdir("/opt/ml/code")
        if len(files_in) > 0:
            content = "/opt/ml/code:\n"
            for f in files_in:
                content += f + "\n"
            write_log(os.path.join(configs["saved_models"], "opt.txt"), content)
            print(content)

    start_time = time.time()
    # unzip data
    unzip_data(configs)
    end_time = time.time()
    took_time = end_time - start_time
    content = str(took_time) + " seconds"
    write_log(os.path.join(configs["saved_models"], "opt.txt"), content)
    print(content)

    # process
    workflow.train(configs)


if __name__ == '__main__':
    main()

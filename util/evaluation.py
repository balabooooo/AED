# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import os
import numpy as np
import copy
import motmetrics as mm
mm.lap.default_solver = 'lap'
import os
from typing import Dict
import numpy as np
import logging
import teta
import pickle

from tao.toolkit.tao import Tao

def teta_eval(ann_file, resfile_path, result_name, tracker_name='my_tracker', print_config=True):
    cats = Tao(ann_file).cats
    # Command line interface:
    default_eval_config = teta.config.get_default_eval_config()
    # print only combined since TrackMAP is undefined for per sequence breakdowns
    default_eval_config["PRINT_ONLY_COMBINED"] = True
    default_eval_config["DISPLAY_LESS_PROGRESS"] = True
    default_eval_config["OUTPUT_TEM_RAW_DATA"] = True
    default_eval_config["NUM_PARALLEL_CORES"] = 8
    default_dataset_config = teta.config.get_default_dataset_config()
    default_dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
    default_dataset_config["GT_FOLDER"] = ann_file
    default_dataset_config["TRACKERS_FOLDER"] = resfile_path
    default_dataset_config["TRACKER_SUB_FOLDER"] = os.path.join(
        resfile_path, result_name
    )
    if not print_config:
        default_eval_config["PRINT_CONFIG"] = False
        default_dataset_config["PRINT_CONFIG"] = False

    evaluator = teta.Evaluator(default_eval_config)
    dataset_list = [teta.datasets.TAO(default_dataset_config)]
    # print("Overall classes performance")
    evaluator.evaluate(dataset_list, [teta.metrics.TETA()])

    eval_results_path = os.path.join(
        resfile_path, tracker_name, "teta_summary_results.pth"
    )
    eval_res = pickle.load(open(eval_results_path, "rb"))

    base_class_synset = set(
        [
            c["name"]
            for c in cats.values()
            if c["frequency"] != "r"
        ]
    )
    novel_class_synset = set(
        [
            c["name"]
            for c in cats.values()
            if c["frequency"] == "r"
        ]
    )

    _compute_teta_on_ovsetup(eval_res, base_class_synset, novel_class_synset)

def _compute_teta_on_ovsetup(teta_res, base_class_names, novel_class_names):
    if "COMBINED_SEQ" in teta_res:
        teta_res = teta_res["COMBINED_SEQ"]

    frequent_teta = []
    rare_teta = []
    for key in teta_res:
        if key in base_class_names:
            frequent_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))
        elif key in novel_class_names:
            rare_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))

    print("Base and Novel classes performance")

    # print the header
    print(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "TETA50",
            "TETA",
            "LocS",
            "AssocS",
            "ClsS",
            "LocRe",
            "LocPr",
            "AssocRe",
            "AssocPr",
            "ClsRe",
            "ClsPr",
        )
    )

    if frequent_teta:
        freq_teta_mean = np.mean(np.stack(frequent_teta), axis=0)

        # print the frequent teta mean
        print("{:<10} ".format("Base"), end="")
        print(*["{:<10.3f}".format(num) for num in freq_teta_mean])

    else:
        print("No Base classes to evaluate!")
        freq_teta_mean = None
    if rare_teta:
        rare_teta_mean = np.mean(np.stack(rare_teta), axis=0)

        # print the rare teta mean
        print("{:<10} ".format("Novel"), end="")
        print(*["{:<10.3f}".format(num) for num in rare_teta_mean])
    else:
        print("No Novel classes to evaluate!")
        rare_teta_mean = None

    return freq_teta_mean, rare_teta_mean
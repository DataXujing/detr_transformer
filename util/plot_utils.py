"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath
import os


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    # if not isinstance(logs, list):
    #     if isinstance(logs, PurePath):
    #         logs = [logs]
    #         print("{} info: logs param expects a list argument, converted to list[Path].".format(func_name))
    #     else:
    #         raise ValueError("{} - invalid argument for logs parameter.\n \
    #         Expect list[Path] or single Path obj, received {}".format(func_name,type(logs)))

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir_ in enumerate([logs]):
        # if not isinstance(dir, PurePath):
        #     raise ValueError("{} - non-Path object in logs argument of {}: \n{}".format(func_name,type(dir),dir))
        # if not dir.exists():
        #     raise ValueError("{} - invalid directory in logs argument:\n{}".format(func_name,dir))
        # # verify log_name exists
        # fn = Path(dir / log_name)
        fn = os.path.join(dir_,log_name)
        # if not fn.exists():
        if not os.path.exists(fn):
            print("-> missing {}.  Have you gotten to Epoch 1 in training?".format(log_name))
            print("--> full path of missing log file: {}".format(fn))
            return

    # load log file(s) and plot
    # dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]
    dfs = [pd.read_json(os.path.join(p,log_name), lines=True) for p in [logs]]
    # print(dfs)


    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=['train_{}'.format(field), 'test_{}'.format(field)],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        # ax.legend([p for p in [logs]])
        ax.legend([field])
        ax.set_title(field)
    # plt.show()
    plt.savefig("log.png")


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError('not supported {}'.format(naming_scheme))
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print('{} {}: mAP@50={}, '.format(naming_scheme,name,round(prec * 100,1)) +
              'score={}, '.format(round(scores.mean(),3)) +
              'f1={}'.format(round(2 * prec * rec / (prec + rec + 1e-8),3))
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs



if __name__ == "__main__":
    log_dir = "./outputs"
    plot_logs(log_dir)
    # python3 util/plot_utils.py













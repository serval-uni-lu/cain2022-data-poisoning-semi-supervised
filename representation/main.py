import pandas as pd
import random
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import random
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, kendalltau
import statistics as sts
from tabulate import tabulate
from icecream import ic


def rq3_table(dataset="mnist"):
    attack_algo_1 = "im"
    attack_algo_2 = "pm"
    attack_algo_3 = "gm"
    root_dir = (
        f"ANONYMISED/output/induction_mlp"
    )

    p_labelled = [0.05, 0.15, 0.25]
    p_poison = [0.05, 0.1, 0.15, 0.2]
    resultat_table = np.empty((3, 13))
    print(f"[{dataset}]", end="\n")
    for i, p_l in enumerate(p_labelled):
        line = np.empty(
            13,
        )
        data_1 = np.loadtxt(f"{root_dir}/lp_{dataset}_{p_l}_{attack_algo_1}_i")
        data_2 = np.loadtxt(f"{root_dir}/lp_{dataset}_{p_l}_{attack_algo_3}_i")
        data_3 = np.loadtxt(f"{root_dir}/lp_{dataset}_{p_l}_{attack_algo_2}_i")
        for count, p_p in enumerate(p_poison):
            line[0] = 1 - data_1[0]
            line[1 + count * 3] = 1 - data_1[1 + count]
            line[2 + count * 3] = 1 - data_2[1 + count]
            line[3 + count * 3] = 1 - data_3[1 + count]

        resultat_table[i] = line

        print(f"[{p_l} labelled] {resultat_table[i][1:]}")

    np.savetxt(
        f"rq3_mlp_{dataset}.csv",
        resultat_table,
        delimiter=",",
        header="0,5_i,5_d,5_s,10_i,10_d,10_s,15_i,15_d,15_s,20_i,20_d,20_s",
    )


def rq3_plot(dataset="mnist"):
    attack_algo_1 = "im"
    attack_algo_2 = "pm"
    root_dir = (
        f"ANONYMISED/output/efficiency/"
    )
    data_1 = np.loadtxt(f"{root_dir}/time_{attack_algo_1}_{dataset}_0.05")
    data_2 = np.loadtxt(f"{root_dir}/time_{attack_algo_2}_{dataset}_0.05")
    data = [data_1, data_2]
    print(
        f"[{dataset}][med][stdev]\nPM:[{sts.median(data_2)}][{sts.stdev(data_2)}]\
        \nIM:[{sts.median(data_1)}][{sts.stdev(data_1)}] "
    )

    plt.boxplot(data)
    plt.savefig(f"efficiency_rq3_{dataset}")
    plt.clf()


def req2_plot_alternative(dataset="rcv1"):
    ssl_algo_1 = "label_propagation"
    ssl_alog_2 = "label_spreading"
    inductive_algo = ["rfc", "mlp"]

    root_dir_rfc = f"ANONYMISED/output/transduction_rfc_output"
    root_dir_mlp = f"ANONYMISED/output/transduction_mlp_output"

    data_y_lp25_rfc = np.loadtxt(f"{root_dir_rfc}/rfc_{ssl_algo_1}_{dataset}_0.25")
    data_y_lp25_mlp = np.loadtxt(f"{root_dir_mlp}/mlp_{ssl_algo_1}_{dataset}_0.25")

    data_y_ls25_rfc = np.loadtxt(f"{root_dir_rfc}/rfc_{ssl_alog_2}_{dataset}_0.25")
    data_y_ls25_mlp = np.loadtxt(f"{root_dir_mlp}/mlp_{ssl_alog_2}_{dataset}_0.25")
    data_x = ["0%", "5%", "10%", "15%", "20%"]

    plt.plot(data_x, 1 - data_y_ls25_rfc, "ro-", label="LS RFC")
    plt.plot(data_x, 1 - data_y_ls25_mlp, "r--", label="LS MLP")
    plt.plot(data_x, 1 - data_y_lp25_rfc, "ko-", label="LP RFC")
    plt.plot(data_x, 1 - data_y_lp25_mlp, "k--", label="LP MLP")
    plt.ylabel("Inductive error rate")
    plt.xlabel("poisoned labels")
    plt.suptitle(f"Induction accuracy on {dataset}")
    plt.legend()
    plt.savefig(f"induction_{dataset}")
    plt.clf()


def rq2_plot(dataset="rcv1"):
    ssl_algo_1 = "label_propagation"
    ssl_alog_2 = "label_spreading"
    inductive_algo = "rfc"
    root_dir = f"ANONYMISED/output/{inductive_algo}_output"
    data_y_lp05 = np.loadtxt(f"{root_dir}/{inductive_algo}_{ssl_algo_1}_{dataset}_0.05")
    data_y_lp15 = np.loadtxt(f"{root_dir}/{inductive_algo}_{ssl_algo_1}_{dataset}_0.15")
    data_y_lp25 = np.loadtxt(f"{root_dir}/{inductive_algo}_{ssl_algo_1}_{dataset}_0.25")
    data_y_ls05 = np.loadtxt(f"{root_dir}/{inductive_algo}_{ssl_alog_2}_{dataset}_0.05")
    data_y_ls15 = np.loadtxt(f"{root_dir}/{inductive_algo}_{ssl_alog_2}_{dataset}_0.15")
    data_y_ls25 = np.loadtxt(f"{root_dir}/{inductive_algo}_{ssl_alog_2}_{dataset}_0.25")
    data_x = ["0%", "5%", "10%", "15%", "20%"]

    plt.plot(data_x, 1 - data_y_lp05, "ko-", label="5% labelled inputs")
    plt.plot(data_x, 1 - data_y_lp15, "k|-", label="15% labelled inputs")
    plt.plot(data_x, 1 - data_y_lp25, "k|--", label="25% labelled inputs")
    plt.plot(data_x, 1 - data_y_ls05, "r-", label="5% labelled inputs")
    plt.plot(data_x, 1 - data_y_ls15, "r|-", label="15% labelled inputs")
    plt.plot(data_x, 1 - data_y_ls25, "r|--", label="25% labelled inputs")
    plt.ylabel("Inductive error rate")
    plt.xlabel("poisoned labels")
    plt.suptitle(f"Induction accuracy on {dataset} with {inductive_algo}")
    plt.legend()
    plt.savefig(f"induction_{dataset}_{inductive_algo}")
    plt.clf()


if __name__ == "__main__":
    datasets = ["mnist", "cifar", "rcv1"]
    # datasets = ["mnist"]
    for ds in datasets:
        # rq2_plot(ds)
        # rq3_plot(ds)
        # rq3_table(ds)
        req2_plot_alternative(ds)
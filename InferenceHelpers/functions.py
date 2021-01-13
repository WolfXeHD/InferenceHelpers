# REMOVE_CELL

import os
import copy
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sps

try:
    plt.style.use('/Users/twolf/.matplotlib/stylelib/myplotstyle.mplstyle')
except:
    pass


def get_data_from_dali(pattern, fetch=False, threshold=False, dali_user="twolf", target="/Users/twolf/Physics/Xenon/NT_projections/MoMa"):
    if threshold:
        splitted_pattern = pattern.split("/")
        splitted_pattern[-2] = splitted_pattern[-2] + "_threshold"
        this_pattern = "/".join(splitted_pattern)
    else:
        this_pattern = pattern

    local_output = this_pattern.split("/")[-2]
    local_output = os.path.join(target, local_output)
    if not os.path.exists(local_output):
        os.makedirs(local_output)
    cmd = f"scp {dali_user}@dali2:{this_pattern} {local_output}"
    print(cmd)
    if fetch:
        os.system(cmd)

    print("data now in -->", local_output)
    return local_output


def get_all_data_from_dali(pattern, fetch=False, dali_user="twolf", target="/Users/twolf/Physics/Xenon/NT_projections/MoMa"):
    local_scan = get_data_from_dali(pattern=pattern, fetch=fetch, threshold=False, dali_user=dali_user, target=target)
    local_thres = get_data_from_dali(pattern=pattern, fetch=fetch, threshold=True, dali_user=dali_user, target=target)
    if fetch:
        print("data fetched!")
    return local_scan, local_thres


def get_fit_files(pattern, show_files=False):
    filelist = sorted(glob.glob(pattern))
    fixed_list = [file for file in filelist if "_fixed.pkl" in file]
    uncond_list = [file.replace("_fixed.pkl", "_uncond.pkl") for file in fixed_list]
    if show_files:
        print(fixed_list)
        print(uncond_list)
    return fixed_list, uncond_list


def get_fit_data(pattern, show_files=False, min_fits=0):
    fixed_files, uncond_files = get_fit_files(pattern, show_files=show_files)

    df_fixed_reses = [pd.read_pickle(file) for file in fixed_files if len(pd.read_pickle(file)) > min_fits]
    df_uncond_reses = [pd.read_pickle(file) for file in uncond_files if len(pd.read_pickle(file)) > min_fits]
    return copy.deepcopy(df_fixed_reses), copy.deepcopy(df_uncond_reses)


def compute_LLRs(fixed, uncond):
    df_fixed = pd.read_pickle(fixed)
    df_uncond = pd.read_pickle(uncond)

    return copy.deepcopy(df_fixed['fval'] - df_uncond['fval'])


def show_available_files(pattern):
    print(pattern)


def plot_llr_distribution(LLR, LL_val, tested_amplitude):
    plt.hist(LLR, bins=200, label="toys with tested amplitude = " + str(tested_amplitude))
    plt.axvline(LL_val, color='red', label=f"median -2 LLR ({LL_val:.3f})")
    plt.yscale('log')
    plt.ylabel("Number of results")
    plt.xlabel("-2 LLR")
    plt.legend()
    plt.show()


def get_single_data_by_pattern(pattern):
    df_fixed, df_uncond = get_fit_data(pattern, show_files=True)
    if len(df_fixed) > 1:
        print(f"{pattern} is ambiguous!")
        return
    elif len(df_fixed) == 0:
        print("No data gotten!")
        return
    else:
        df_fixed = df_fixed[0]
        df_uncond = df_uncond[0]
    return df_fixed, df_uncond


def plot_single_llr_by_pattern(pattern, plot=True):
    df_fixed, df_uncond = get_single_data_by_pattern(pattern)
    df_fixed, df_uncond = remove_ill_fits(df_fixed, df_uncond)
    LLR = df_fixed['fval'] - df_uncond['fval']
    LL_val = np.quantile(LLR, 0.5)
    tested_amplitude = np.mean(df_fixed['A_0vbb'])
    if plot:
        plot_llr_distribution(LLR, LL_val, tested_amplitude)


def remove_ill_fits(df_fixed, df_uncond):
    mask = df_fixed["acceptance_0vbb"] > 5
    return df_fixed[mask], df_uncond[mask]


def aggreate_LLRs(pattern, quantile, n_plot=0, min_fits=0, remove_fits=False):
    df_fixed_reses, df_uncond_reses = get_fit_data(pattern, min_fits=min_fits)
    amplitude = []
    LL_scan = []
    for idx, (fixed_res, uncond_res) in enumerate(zip(df_fixed_reses, df_uncond_reses)):
        if len(fixed_res) == 0:
            continue

        if remove_fits:
            df_fixed, df_uncond = remove_ill_fits(fixed_res, uncond_res)
        else:
            df_fixed, df_uncond = fixed_res, uncond_res

        LLR = df_fixed["fval"] - df_uncond["fval"]
        tested_amplitude = np.mean(df_fixed["A_0vbb"])
        # LLR = np.where(LLR < 0, 0, LLR)

        LL_val = np.quantile(LLR, quantile)
        amplitude.append(tested_amplitude)
        LL_scan.append(LL_val)

        if idx < n_plot and quantile == 0.5:
            plot_llr_distribution(LLR, LL_val, tested_amplitude)

        if idx < n_plot and quantile == 0.9:
            xbins = np.linspace(0, 10, 1000)
            plt.hist(LLR, bins=1000, density=True, cumulative=-1, label="toys with amplitude = " + str(tested_amplitude))
            plt.plot(xbins, sps.chi2.sf(xbins, df=1), label="asymptotic $\chi^2$ survival function")
            plt.axvline(np.quantile(LLR, quantile), color='red')
            plt.axvline(sps.chi2.ppf(quantile, df=1), color='red', dashes=(5, 2.5))
            plt.yscale("log")
#             plt.xlim(0, 10)
            plt.ylabel("Number of results")
            plt.xlabel("-2 LLR")
            plt.legend()
            plt.show()

    df_LL_scan = pd.DataFrame([LL_scan, amplitude], ["LL_scan", "amplitude"]).T
    df_LL_scan.sort_values(by=['amplitude'], inplace=True)
    if len(df_LL_scan) == 0:
        print(f"No datapoints gotten for {pattern}")
    return df_LL_scan


def plot_likelihood_scan(local_path, plotopt={'linestyle': 'None', 'marker': 'o'}, plot_thres=False, min_fits=0, remove_fits=False):
    df_LL_scan = aggreate_LLRs(local_path + "/*pkl", quantile=0.5, n_plot=0, min_fits=min_fits, remove_fits=remove_fits)
    plt.xlabel("tested amplitude")
    plt.ylabel("-2 LLR")
    threshold = sps.chi2.ppf(0.9, df=1)

    if plot_thres:
        plt.axhline(threshold, color='red', label="threshold asymptotic")
    plt.plot(df_LL_scan["amplitude"], df_LL_scan["LL_scan"], **plotopt)

    plt.title("Likelihood scan")
    #  plt.legend()

    #  mask = df_LL_scan["LL_scan"] <= threshold
    #  max_id = df_LL_scan[mask]["amplitude"].idxmax()
    # plt.xlim(0, 0.0006)
    # plt.ylim(0, 1)
    # threshold_amplitude = df_LL_scan.loc[max_id]["amplitude"]
    # plt.axvline(threshold_amplitude, color='blue', label="amplitude crossing")
    # plt.show()
    return df_LL_scan


def plot_threshold(local_path, plotopt={'linestyle': 'None', 'marker': 'o'}, min_fits=0, remove_fits=False):
    df_LL_threshold = aggreate_LLRs(local_path + "/*pkl", quantile=0.9, n_plot=0, min_fits=min_fits, remove_fits=remove_fits)
#     plt.xlabel("tested amplitude")
#     plt.ylabel("-2 LLR")
    #  threshold = sps.chi2.ppf(0.9, df=1)

    plt.plot(df_LL_threshold["amplitude"], df_LL_threshold["LL_scan"], **plotopt)

    # plt.title("Likelihood scan")
    #  plt.legend()
    return df_LL_threshold


def derive_columns(df_uncond):
    cols = [col for col in df_uncond.columns if not col.startswith("s_")]  # remove starting values from fit
    cols = [col for col in cols if not col.startswith("toy_")]  # remove toy-values from fit
    return cols


def check_fitting_details(pattern, col=None, plot=True, remove_fits=False):
    print(pattern)
    df_fixed, df_uncond = get_single_data_by_pattern(pattern)
    if remove_fits:
        df_fixed, df_uncond = remove_ill_fits(df_fixed, df_uncond)
    LLR = df_fixed["fval"] - df_uncond["fval"]
    cols = derive_columns(df_uncond)

    if plot:
        plot_2D_distributions(df_uncond, df_fixed, LLR, col, cols)

    return df_fixed, df_uncond


def plot_2D_distributions(df_uncond, df_fixed, LLR, col, cols):
    if col is None:
        for col in cols:
            plt.scatter(LLR, df_uncond[col], s=20, label="unconditional")
            plt.scatter(LLR, df_fixed[col], s=10, label="fixed")
            plt.xlabel("-2 LLR")
            plt.ylabel(col)
            toy_col = "toy_" + col
            if toy_col in df_uncond.keys():
                plt.axhline(df_uncond[toy_col].mean(), color='red', label="toy value")
            plt.legend()
            plt.show()
    else:
        if col not in cols:
            print("col:", col, "does not exist!")
            return
        plt.scatter(LLR, df_uncond[col], s=20, label="unconditional")
        plt.scatter(LLR, df_fixed[col], s=10, label="fixed")
        plt.xlabel("-2 LLR")
        plt.ylabel(col)
        toy_col = "toy_" + col
        if toy_col in df_uncond.keys():
            plt.axhline(df_uncond[toy_col].mean(), color='red', label="toy value")
        plt.legend()
        plt.show()


def plot_1D_distributions(df_uncond, df_fixed, cols, col=None):
    if col is None:
        cols_to_iterate = cols
    else:
        cols_to_iterate = [col]

    if cols_to_iterate[0] not in cols:
        print("Column", cols_to_iterate[0], "not found!")
        raise SystemExit

    for this_col in cols_to_iterate:
        toy_col = "toy_" + this_col
        if toy_col in df_uncond.columns:
            print(toy_col, df_uncond[toy_col].mean())
            if this_col in cols:
                plt.hist(df_uncond[this_col], bins=200, label="unconditional fit")
                plt.hist(df_fixed[this_col], bins=200, alpha=0.5, label="fixed fit")

                plt.xlabel(this_col)
                plt.axvline(df_uncond[this_col].mean(), color='red', label="toy value")
                plt.legend()
                plt.show()

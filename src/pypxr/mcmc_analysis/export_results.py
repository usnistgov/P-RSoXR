"""

Generic input file for running PyPXR modeling
Updated 08/05/2021 For GitHub


"""
# Basic imports Items -
import os.path
import pickle
import sys

import corner
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from refnx.analysis import GlobalObjective, process_chain
from typing_extensions import deprecated

# Import PyPXR
try:  # Check for pip install
    from pypxr.reflectivity import PXR_ReflectModel
except ImportError:
    from reflectivity import PXR_ReflectModel
"""



"""


# This is the function you want --
def export_mcmc_summary(
    save_path,
    save_name,
    fitter,
    burn_thresh=0.005,
    convergence_thresh=0.95,
    numpnts=512000,
):
    objective = fitter.objective
    current_chain = fitter.chain
    sampler = fitter.sampler
    logpost = fitter.logpost
    mask = np.full((logpost.shape[1]), True)  # generate mask to make sure it exists
    # Construct a new objective to apply postprocessing
    if hasattr(objective, "objectives"):
        objective_processed = objective
    else:
        objective_processed = GlobalObjective([objective])

    # Burn and thin chain to final distribution:
    chainmin_lp = np.max(
        logpost, axis=0
    )  # Find the best fit for each chain - individual chain minima
    globalmin_lp = np.max(logpost)  # Global minima of the fit
    numchain = np.shape(logpost)[1]  # [0 - draws] [1 - Chains]

    for nburn, array in enumerate(
        logpost
    ):  # Cycle through the draws to find if chain converges
        converge = (
            np.abs(array - chainmin_lp) / np.abs(chainmin_lp) < burn_thresh
        )  # How close is each chain to the minima
        numpass = np.sum(converge)  # How many have converged at this point
        if numpass / numchain > convergence_thresh:  # Have enough converged?
            mask = (
                np.abs(array - globalmin_lp) / np.abs(globalmin_lp) < burn_thresh
            )  # Throw away chains that don't reach the minima
            break

    totalpoints = np.size(logpost[nburn:, mask])  # Total points not burned
    thinpnts = round(
        totalpoints / numpnts
    )  # Calculate the number to thin based on desired points
    nthin = (
        thinpnts if thinpnts > 0 else 1
    )  # Don't thin if you don't have enough points

    # Reprocess the chain with the appropriate
    processed_chain = process_chain(
        objective_processed,
        current_chain[:, mask, :],
        nburn=nburn,
        nthin=nthin,
        flatchain=True,
    )

    with open(os.path.join(save_path + save_name + "_fitter.pkl"), "wb+") as f:
        pickle.dump(fitter, f)
    with open(os.path.join(save_path + save_name + "_obj_processed.pkl"), "wb+") as f:
        pickle.dump(objective_processed, f)

    save_logpost(path=save_path, fitter=fitter)
    save_chains(
        path=save_path, objective=objective_processed, chain_processed=processed_chain
    )
    save_fitresults(path=save_path, objective=objective_processed, save_name=save_name)
    save_PSoXR(path=save_path, objective=objective_processed, save_name=save_name)
    save_depthprofile(
        path=save_path, objective=objective_processed, save_name=save_name
    )
    save_corner(path=save_path, objective=objective_processed)


def save_logpost(path, fitter):
    lp = fitter.logpost
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    ax.plot(np.abs(lp), alpha=0.5)
    ax.set_ylabel("logpost", fontsize=10)
    ax.set_xlabel("Generation [#]")
    plt.savefig(os.path.join(path, "logpost.png"), dpi=100)
    plt.close()


def save_chains(path, objective, chain_processed):
    # Make a folder to save each chain plot
    savepath = path + "chain_stats/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    labels = objective.varying_parameters().names()
    num = len(labels)

    for i in range(num):
        fig, ax = plt.subplots(1, 1, figsize=(4, 2), constrained_layout=True)
        temp_chain = chain_processed[i].chain
        ax.plot(temp_chain, alpha=0.3)
        ax.set_ylabel(labels[i], fontsize=10)
        ax.set_xlabel("Generation [#]", fontsize=10)
        plt.savefig(os.path.join(savepath + labels[i] + ".png"), dpi=100)
        plt.close()


def save_fitresults(path, objective, save_name):
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    filename = save_name + "_Results.xlsx"
    with pd.ExcelWriter(os.path.join(path + filename)) as writer:
        for objective in objective.objectives:
            name = str(objective.name)
            params_with_chain = [
                p for p in objective.parameters.flattened() if p.chain is not None
            ]
            cols = ["value", "std", "vary", "logp", "norm_mu", "norm_std", "prior_pdf"]
            out = pd.DataFrame(columns=cols)
            index_list = []
            hist_header = {}
            hist_popdensity = {}

            corrDF = pd.DataFrame()

            for param in params_with_chain:
                index_list.append(param.name)
                value = param.value
                std = param.stderr
                vary = param.vary
                logp = param.logp()
                mu, std = norm.fit(param.chain)
                prior_pdf = 1  # 1/(param.bounds.ub - param.bounds.lb)

                out = out.append(
                    pd.DataFrame(
                        [[value, std, vary, logp, mu, std, prior_pdf]], columns=cols
                    )
                )

                # Build Histogram for Posterior
                vals, bins = np.histogram(param.chain, bins=75, density=True)
                bin_min = bins[0]
                bin_max = bins[-1]
                bin_diff = bins[1] - bins[0]
                hist_header[param.name] = [bin_min, bin_max, bin_diff]
                hist_popdensity[param.name] = vals

                # Grab variable parameters to calculate fit correlations
                if vary:
                    corrDF[param.name] = param.chain

            # Save results
            out.index = index_list
            out.to_excel(writer, sheet_name=(name + "_Values"))

            # Save Histogram
            hist_stats = pd.DataFrame(hist_header)
            hist_stats.index = ["bin_min", "bin_max", "bin_diff"]
            hist_vals = pd.DataFrame(hist_popdensity)
            hist_output = hist_stats.append(hist_vals, ignore_index=False)
            hist_output.to_excel(writer, sheet_name=(name + "_Hist"))

            # Save Correlation
            corr = corrDF.corr(method="pearson").round(decimals=5)
            corr.to_excel(writer, sheet_name=(name + "_Correlation"))
            corr_plot = corrDF.corr(method="pearson").round(decimals=1)
            save_correlation_diagram(path, corr_plot, name)


def save_PSoXR(path, objective, save_name):
    import pandas as pd

    filename = save_name + "_Profs.xlsx"

    with pd.ExcelWriter(os.path.join(path + filename)) as writer:
        for i, objective in enumerate(objective.objectives):
            # Model information
            name = str(objective.name)
            energy = objective.model.energy
            energy_name = str(energy).replace(".", "pt")
            structure = objective.model.structure
            scale = objective.model.scale.value
            bkg = objective.model.bkg.value
            dq = objective.model.dq.value
            offset = objective.model.q_offset.value
            datapol = list(
                objective.model.pol
            )  # Check polarization associated with objective

            # Possible Polarizations and color scheme
            pol_list = ["s", "p"]
            pol_color = ["#FE6100", "#785EF0"]

            data = objective.data  # Get data
            res = objective.residuals()  # Calculate residuals

            # Organize based on pol
            if len(datapol) == 2:  # Is it a concatenated model?
                init_loc = [0, np.argmax(np.abs(np.diff(data.data[0]))) + 1]
                swap_loc = [
                    np.argmax(np.abs(np.diff(data.data[0]))) + 1,
                    -1,
                ]  # Run up until the last value in the
            else:
                init_loc = [0, 0]
                swap_loc = [-1, -1]

            # Cycle through objective to save results
            for i, pol in enumerate(pol_list):  # Possible polarizaitons
                if pol in datapol:
                    polname = str(pol + "pol")
                    model = PXR_ReflectModel(
                        structure,
                        scale=scale,
                        bkg=bkg,
                        dq=dq,
                        energy=energy,
                        pol=pol,
                        name=(name + "_" + polname),
                    )
                    data_qvals = data.data[0][init_loc[i] : swap_loc[i]]
                    data_refl = data.data[1][init_loc[i] : swap_loc[i]]
                    data_err = data.data[2][init_loc[i] : swap_loc[i]]

                    # Calculate the model values
                    model_refl = model.model(data_qvals + offset)
                    model_res = res[init_loc[i] : swap_loc[i]]

                    # Interpolate for a better plot
                    lowq = np.amin(data_qvals)
                    highq = np.amax(data_qvals)
                    prettymodel_qvals = np.linspace(lowq, highq, 1000) + offset
                    prettymodel_refl = model.model(prettymodel_qvals)

                    sheetname = name + "_" + energy_name + "_" + polname
                    cols = [
                        "qvals",
                        str("refl_" + pol + "pol"),
                        str("reflu_" + pol + "pol"),
                        str("model_" + pol + "pol"),
                        str("res_" + pol + "pol"),
                    ]
                    cols_interp = ["qvals_interp", "model_" + pol + "pol"]
                    output = pd.DataFrame(
                        np.array(
                            [data_qvals, data_refl, data_err, model_refl, model_res]
                        ).T,
                        columns=cols,
                    )
                    output_interp = pd.DataFrame(
                        np.array([prettymodel_qvals, prettymodel_refl]).T,
                        columns=cols_interp,
                    )

                    # Save results to spreadsheet
                    output.to_excel(writer, sheet_name=(sheetname + ".xlsx"))
                    output_interp.to_excel(
                        writer, sheet_name=(sheetname + "_interp.xlsx")
                    )

                    # Save plots
                    # Make Figure
                    fig = plt.figure(constrained_layout=True, figsize=(4, 3))
                    fig.suptitle(
                        name + "_" + energy_name + "_" + pol + "pol", fontsize=10
                    )
                    grid = fig.add_gridspec(5, 5)
                    refl_ax = fig.add_subplot(grid[1:, :])
                    res_ax = fig.add_subplot(grid[0, :], sharex=refl_ax)

                    # Append Data
                    res_ax.scatter(
                        data_qvals,
                        model_res,
                        color="w",
                        marker="o",
                        s=3,
                        edgecolors=pol_color[i],
                        linewidth=0.5,
                        label=str(pol + "pol"),
                    )
                    refl_ax.errorbar(
                        data_qvals,
                        data_refl,
                        data_err,
                        color=pol_color[i],
                        marker="o",
                        ms=3,
                        ls="",
                        markerfacecolor="w",
                        markeredgewidth=0.5,
                        zorder=10,
                        label=str(pol + "pol"),
                    )
                    refl_ax.plot(
                        data_qvals, model_refl, color="0", lw=1, zorder=30, label="fit"
                    )

                    # Residual graph settings
                    res_ax.set_ylabel("Res [%]", fontsize=10)
                    res_ax.tick_params(axis="both", labelsize=10)
                    res_ax.set_ylim(-10, 10)
                    plt.setp(res_ax.get_xticklabels(), visible=False)

                    # Profile graph settings
                    refl_ax.legend(fontsize=10)
                    refl_ax.set_yscale("log")
                    refl_ax.set_ylabel("Reflectivity", fontsize=10)
                    refl_ax.set_xlabel("q [$\AA^{-1}$]", fontsize=10)
                    refl_ax.tick_params(axis="both", labelsize=10)

                    # Save Figure
                    plt.savefig(
                        os.path.join(
                            path, name + "_" + energy_name + "_" + pol + "pol" + ".png"
                        ),
                        dpi=200,
                    )
                    plt.close()


def save_depthprofile(path, objective, save_name):
    import pandas as pd

    filename = save_name + "_DepthProfiles.xlsx"
    with pd.ExcelWriter(os.path.join(path + filename)) as writer:
        for i, objective in enumerate(objective.objectives):
            name = str(objective.name)
            structurename = objective.model.structure.name

            savename = str(name + "_" + structurename + "_" + "_depthprofile.csv")

            datapol = [
                objective.model.pol
            ]  # Check what polarization is associated with the objective
            zed, prof = objective.model.structure.sld_profile(
                align=-1
            )  # calculate SLD profile
            iso_prof = prof.sum(axis=1) / 3
            diff_prof = prof[:, 0] - prof[:, 2]

            # Compile components for dataframe
            depthprofile = {
                "zed": -1 * zed,
                "delta_trace": np.real(iso_prof),
                "beta_trace": np.imag(iso_prof),
                "birefringence": np.real(diff_prof),
                "dichroism": np.imag(diff_prof),
                "dxx": np.real(prof[:, 0]),
                "bxx": np.imag(prof[:, 0]),
                "dyy": np.real(prof[:, 1]),
                "byy": np.imag(prof[:, 1]),
                "dzz": np.real(prof[:, 2]),
                "bzz": np.imag(prof[:, 1]),
            }
            df = pd.DataFrame(depthprofile)

            # Save to spreadsheet
            df.to_excel(writer, sheet_name=(structurename + "_prof"))

            # Plot results
            fig_depthprofile, ax_dp = plt.subplots(1, 1, figsize=(4, 3))
            df.plot(ax=ax_dp, x="zed", y="delta_trace", color=["#0072B2"], lw=1.5)
            df.plot(ax=ax_dp, x="zed", y="beta_trace", color=["#D55E00"], lw=1.5)

            df.plot(
                ax=ax_dp,
                x="zed",
                y="dxx",
                style=[":"],
                color=["#0072B2"],
                lw=1,
                legend=False,
            )
            df.plot(
                ax=ax_dp,
                x="zed",
                y="dzz",
                style=["--"],
                color=["#0072B2"],
                lw=1,
                legend=False,
            )
            df.plot(
                ax=ax_dp,
                x="zed",
                y="bxx",
                style=[":"],
                color=["#D55E00"],
                lw=1,
                legend=False,
            )
            df.plot(
                ax=ax_dp,
                x="zed",
                y="bzz",
                style=["--"],
                color=["#D55E00"],
                lw=1,
                legend=False,
            )

            ax_dp.set_ylabel("Index of Refraction")
            ax_dp.set_xlabel("Distance from substrate [A]")
            plt.legend(["delta", "beta"])
            plt.savefig(os.path.join(path, name + "_DepthProfile.png"), dpi=200)
            plt.close()

            fig_orientprofile, ax_op = plt.subplots(1, 1, figsize=(4, 3))
            df.plot(ax=ax_op, x="zed", y="birefringence", color=["#0072B2"])
            df.plot(ax=ax_op, x="zed", y="dichroism", color=["#D55E00"])

            ax_op.set_xlabel("Distance from substrate [A]")
            plt.savefig(os.path.join(path, name + "_OrientationProfile.png"), dpi=200)
            plt.close()


def save_corner(path, objective):
    for i, objective in enumerate(objective.objectives):
        name = str(objective.name)
        labels = objective.varying_parameters().names()
        num = len(labels)
        fig = objective.corner(bins=50, showtitles=True)
        axes = np.array(fig.axes).reshape((num, num))

        # Loop over the diagonal
        for i in range(num):
            ax = axes[i, i]
            ax.yaxis.set_label_coords(-0.3, 0.5)
            ax.xaxis.set_label_coords(0.5, -0.4)
            ax.xaxis.label.set_size(12)
            ax.tick_params(axis="both", which="major", labelsize=12)

        for yi in range(num):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.yaxis.label.set_size(12)
                ax.yaxis.set_label_coords(-0.4, 0.5)
                ax.xaxis.label.set_size(12)
                ax.xaxis.set_label_coords(0.5, -0.4)
                ax.tick_params(axis="both", which="major", labelsize=12)

        plt.savefig(os.path.join(path, "corner_" + name + ".png"), dpi=100)
        plt.close()


@deprecated("Function `save_integratedtime` is deprecated and will throw an exception")
def save_integratedtime(objective, input_chain, name="auto_correlation_time"):
    import numpy as np

    list_name = objective.varying_parameters().names()

    labels = np.full(len(list_name), "", dtype=object)
    auto = np.full(len(list_name), 0, dtype=float)

    for i in range(len(list_name)):
        labels[i] = list_name[i]
        temp_chain = input_chain[i].chain
        auto[i] = integrated_time(temp_chain[i], quiet=True)

    out = np.rollaxis(np.array([labels, auto]), 1, 0)
    np.savetxt(name + ".txt", out, fmt="%s, %f", delimiter=",")


def save_correlation_diagram(path, corr, name):
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.heatmap(
        corr.round(decimals=1),
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(240, 10, n=200),
        square=True,
        linewidth=1.5,
        annot=True,
        ax=ax,
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.6},
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize=12
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    # Got this from stackexchange...works by changing fontsize
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(os.path.join(path, name + "_corr_plot.png"), dpi=200)
    plt.close()

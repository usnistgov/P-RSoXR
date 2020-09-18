
# start off with the necessary imports
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 16})
import corner
import pickle

#required modules from refnx
from refnx.analysis import Transform, CurveFitter, Objective, GlobalObjective, Parameter, integrated_time
from refnx.analysis import process_chain, load_chain, autocorrelation_chain
from refnx.ani_reflect.ani_reflect_model import ani_ReflectModel



###This is the function you want -- 
def postprocess_fit(objective, nburn=0, nthin=1):

    #Build new objective to postprocess the chain
    if hasattr(objective, 'objectives'):
        objective_processed = objective
    else:
        objective_processed = GlobalObjective([objective])

    current_chain = load_chain('SaveChain.txt') ##Assuming that you are currently in the folder that this data is stored
    processed_chain = process_chain(objective_processed, current_chain, nburn=nburn, nthin=nthin)
    
    with open("objective_processed.txt", 'wb+') as f:
        pickle.dump(objective_processed, f)
    
    plot_MCMC(objective_processed, processed_chain)
    save_integratedtime(objective_processed, processed_chain)
    plot_corner(objective_processed)
    plot_PSoXR(objective_processed)
    save_fitresults(objective_processed)
    
    

def plot_MCMC(objective, input_chain, name='MCMC_Chain_process'):
    labels = objective.varying_parameters().names()
    num = len(labels)
    chain_fig, chain_axes = plt.subplots(num, figsize=(10, 14), sharex=True)
    for i in range(num):
        ax = chain_axes[i]
        temp_chain = input_chain[i].chain
        ax.plot(temp_chain, alpha=0.3)
        ax.set_xlim(0, len(temp_chain))
        ax.set_ylabel(labels[i],rotation=0,fontsize=12)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax.tick_params(axis='y',labelsize=12)
        ax.yaxis.offsetText.set_fontsize(12)
        ax.yaxis.set_label_coords(-0.15, 0.5)
    #ax.yaxis.set_label_coords(-0.15, 0.5)
    chain_axes[num-1].set_xlabel('Generation [#]')
    
    plt.savefig(name + '.png', dpi=100)
    plt.close()

def plot_PSoXR(objective,qlow=0.006,qhigh=0.26):

    for i, objective in enumerate(objective.objectives):
    
        name = str(objective.name)
        energy = objective.model.energy
        structure = objective.model.structure
        
        fit_qvals = np.linspace(qlow,qhigh,1000)
        fit_spol = ani_ReflectModel(structure, scale=1,bkg=0,dq=0,energy=energy,pol='s',name='spol')
        fit_ppol = ani_ReflectModel(structure, scale=1,bkg=0,dq=0,energy=energy,pol='p',name='spol')
        
        data = objective.data
        res = objective.residuals()
        
        ##Split s and p pol
        swap_loc = np.argmax(np.abs(np.diff(data.data[0]))) ##Where does it swap from the maximum Q of spol to the minimum Q at ppol
        
        spol_qvals = data.data[0][:swap_loc+1]
        spol_data = data.data[1][:swap_loc+1]
        spol_u = data.data[2][:swap_loc+1]
        spol_res = res[:swap_loc+1]
        
        ppol_qvals = data.data[0][swap_loc+1:]
        ppol_data = data.data[1][swap_loc+1:]
        ppol_u = data.data[2][swap_loc+1:]
        ppol_res = res[swap_loc+1:]

        ##Plot stuff
        fig = plt.figure(constrained_layout=True,figsize=(4,3))
        grid = fig.add_gridspec(5, 5)
        refl_ax = fig.add_subplot(grid[1:,:])
        res_ax = fig.add_subplot(grid[0,:],sharex=refl_ax)
        
        res_spol = res_ax.scatter(spol_qvals, spol_res,color='w',marker='o',s=3, edgecolors='#FE6100', linewidth=0.5, label='s-pol')
        res_ppol = res_ax.scatter(ppol_qvals, ppol_res,color='w',marker='o',s=3, edgecolors='#785EF0', linewidth=0.5, label='p-pol')
        
        res_ax.set_ylabel('Res [%]',fontsize=8)
        res_ax.tick_params(axis='both',labelsize=8)
        res_ax.set_ylim(-10,10)
        plt.setp(res_ax.get_xticklabels(), visible=False)


        spol_data_graph = refl_ax.errorbar(spol_qvals, spol_data, spol_u, color='#FE6100', marker='o', ms=3, ls="", markerfacecolor='w', markeredgewidth=0.5, zorder=10, label='s-pol')
        ppol_data_graph = refl_ax.errorbar(ppol_qvals, ppol_data, ppol_u, color='#785EF0', marker='o', ms=3, ls="", markerfacecolor='w', markeredgewidth=0.5, zorder=10, label='p-pol')
        fit_spol_graph = refl_ax.plot(fit_qvals, fit_spol.model(fit_qvals), color='0', lw=1,zorder=30, label='fit')
        fit_ppol_graph = refl_ax.plot(fit_qvals, fit_ppol.model(fit_qvals), color='0', lw=1,zorder=30)
        
        refl_ax.legend(fontsize=8)
        refl_ax.set_yscale('log')
        refl_ax.set_ylabel('Reflectivity',fontsize=8)
        refl_ax.set_xlabel('q [$\AA^{-1}$]',fontsize=8)
        refl_ax.tick_params(axis='both',labelsize=8)

        plt.savefig('Profile_Fits_' + name + '.png',dpi=200)
        plt.close()
        np.savetxt('model_qvals.txt',fit_qvals)
        np.savetxt(('spol_fit_en'+str(i)+'.txt'),fit_spol.model(fit_qvals)) 
        np.savetxt(('ppol_fit_en'+str(i)+'.txt'),fit_ppol.model(fit_qvals)) 
 
        
def plot_corner(objective):
    labels = objective.varying_parameters().names()
    num = len(labels)
    fig = objective.corner(bins=50,showtitles=True)
    axes = np.array(fig.axes).reshape((num, num))

    # Loop over the diagonal
    for i in range(num):
        ax = axes[i, i]
        ax.yaxis.set_label_coords(-0.3,0.5)
        ax.xaxis.set_label_coords(0.5,-0.4)
        ax.xaxis.label.set_size(10)
        ax.tick_params(axis='both', which='major', labelsize=10)


    for yi in range(num):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.yaxis.label.set_size(10)
            ax.yaxis.set_label_coords(-0.4,0.5)
            ax.xaxis.label.set_size(10)
            ax.xaxis.set_label_coords(0.5,-0.4)
            ax.tick_params(axis='both', which='major', labelsize=10)

        
    plt.savefig('corner_process.png',dpi=100)
    plt.close()
    
def plot_psoxr_corner(objective):
    
    important_params = ['thick', 'rough']    
    
    for objective in objective.objectives:
        labels = objective.parameters().names()




def save_integratedtime(objective, input_chain, name='auto_correlation_time'):
    import numpy as np
    list_name = objective.varying_parameters().names()
    
    labels = np.full(len(list_name) ,'',dtype=object)
    auto = np.full(len(list_name) ,0 ,dtype=float)

    
    for i in range(len(list_name)):
        labels[i] = list_name[i]
        temp_chain = input_chain[i].chain
        auto[i] = integrated_time(temp_chain[i],quiet=True)
    
    out = np.rollaxis(np.array([labels,auto]),1,0)
    np.savetxt(name + '.txt',out,fmt="%s, %f",delimiter=',')
    

def save_fitresults(objective, name='FitResults'):
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    
    for objective in objective.objectives:
        
        objective_name = str(objective.name)
        
        params_with_chain = [
            p for p in objective.parameters.flattened() if p.chain is not None
        ]
        save_columns = ['value', 'stderr', 'vary', 'norm_mu', 'norm_std']
        save_output = pd.DataFrame(columns=save_columns)
        index_list = []

        hist_header_info = {}
        hist_popdensity = {}
        
        corrDF = pd.DataFrame()
        
        for param in params_with_chain:
            index_list.append(param.name)
            value = param.value
            stderr = param.stderr
            vary = param.vary
            mu, std = norm.fit(param.chain)
            
            save_output = save_output.append(pd.DataFrame([[value, stderr, vary, mu, std]], columns=save_columns))

            vals, bins = np.histogram(param.chain, bins=75, density=True)
            bin_min = bins[0]
            bin_max = bins[-1]
            bin_diff = bins[1] - bins[0]
            
            hist_header_info[param.name] = [bin_min, bin_max, bin_diff]
            hist_popdensity[param.name] = vals
            if vary or 'rough' in param.name or 'thick' in param.name:
                corrDF[param.name] = param.chain
        #Save Results
        save_output.index = index_list
        save_output.to_csv((objective_name+'_FitResults.csv'))
        #Save Histogram
        hist_stats = pd.DataFrame(hist_header_info)
        hist_stats.index = ['bin_min', 'bin_max', 'bin_diff']
        hist_vals = pd.DataFrame(hist_popdensity)
        hist_output = hist_stats.append(hist_vals, ignore_index=False)
        hist_output.to_csv((objective_name+'_HistogramResults.csv'))
        #Save Correlation
        corr = corrDF.corr(method='pearson').round(decimals=1)
        corr.to_csv((objective_name + '_correlations.csv'))
        save_correlation_diagram(corr, objective_name)
        
    return 0
    
def save_correlation_diagram(corr, name):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(
        corr.round(decimals=1), 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(240, 10, n=200),
        square=True,
        linewidth=1.5,
        annot=True,
        ax=ax,
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.6}
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        fontsize=10
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=10
    )
    #Got this from stackexchange...works by changing fontsize
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    
    plt.savefig((name + '_corr_plot.png'),dpi=200)
    plt.close()
    
def save_fitresults_old(objective, name='FitResults'):
    import numpy as np
    
    for objective in objective.objectives:
    
        Param_list = objective.parameters.varying_parameters()
        Constraint_list = objective.parameters.constrained_parameters()
    
        numParams = len(Param_list)
        numConstraints = len(Constraint_list)
    
        Param_names = np.full(numParams,'',dtype=object)
        Param_vals = np.full(numParams,0,dtype=float)
        Param_stderr = np.full(numParams,0,dtype=float)
    
        Constraint_names = np.full(numConstraints,'',dtype=object)
        Constraint_vals = np.full(numConstraints,0,dtype=float)
        Constraint_stderr = np.full(numConstraints,0,dtype=float) 
    
    #cycle through the parameters to assign them 
        for i in range(numParams):
            Param_names[i] = str(Param_list[i].name)
            Param_vals[i] = Param_list[i].value
            Param_stderr[i] = Param_list[i].stderr
    
        for i in range(numConstraints):
            Constraint_names[i] = str(Constraint_list[i].name)
            Constraint_vals[i] = Constraint_list[i].value
            Constraint_stderr[i] = Constraint_list[i].stderr
        
        Param_results = np.rollaxis(np.array([Param_names,Param_vals,Param_stderr]),1,0)
        Constraint_results = np.rollaxis(np.array([Constraint_names,Constraint_vals,Constraint_stderr]),1,0)
        np.savetxt('objective_'+ str(objective.name) + '_vary_param.txt',Param_results,fmt="%s, %f, %f",delimiter=',')
        np.savetxt('objective_'+ str(objective.name) + '_constraint_param.txt',Constraint_results,fmt="%s, %f, %f",delimiter=',')
    
    
    


def custom_corner(objective, vary_num=[], constraint_num=[], **kwds):
    
    var_pars = [objective.parameters.varying_parameters()[i] for i in vary_num]
    #cons_pars = [objective.parameters.constrained_parameters()[j] for j in constraint_num]
    chain = np.array([par.chain for par in var_pars])#, [con.chain for con in cons_pars]])
    chain = chain.reshape(len(chain),-1).T
    kwds["labels"] = labels
    kwds["quantiles"] = [0.16,0.5,0.84]
    return corner.corner(chain, **kwds)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

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



###This is the function you want -- 
def postprocess_fit(nburn=0, nthin=1):

    #Build a new objective to reload data and keep the previous one the same
    objective_post_en1 = Objective(model_en1, data_en1, transform = Transform('logY'), name = 'en1')
    objective_post_en2 = Objective(model_en2, data_en2, transform = Transform('logY'), name = 'en2')
    objective_processed = GlobalObjective([objective_post_en1, objective_post_en2])

    current_chain = load_chain('SaveChain.txt') ##Assuming that you are in the folder that this data is in
    processed_chain = process_chain(objective_processed, current_chain, nburn=nburn, nthin=nthin)
    
    plot_MCMC(objective_processed, processed_chain)
    save_integratedtime(objective_processed, processed_chain)
    plot_corner(objective_processed)
    plot_PSoXR(objective_processed)
    save_fitresults(objective_processed)
    
    
    


def plot_MCMC(objective, chain, name='MCMC_Chain'):
    labels = objective.varying_parameters().names()
    num = len(labels)
    chain_fig, chain_axes = plt.subplots(num, figsize=(10, 14), sharex=True)
    
    for i in range(num):
        ax = chain_axes[i]
        temp_chain = chain[i].chain
        ax.plot(chain, alpha=0.3)
        ax.set_xlim(0, len(chain))
        ax.set_ylabel(labels[i],rotation=0,fontsize=12)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax.tick_params(axis='y',labelsize=12)
        ax.yaxis.offsetText.set_fontsize(12)
        ax.yaxis.set_label_coords(-0.15, 0.5)
    #ax.yaxis.set_label_coords(-0.15, 0.5)
    chain_axes[num-1].set_xlabel('Generation [#]')
    
    plt.savefig(name + '.png', dpi=300)
    plt.close()

def plot_PSoXR(objective)

    for objective in objective.objectives:
    
        name = objective.name
        energy = objective.model.Energy
        structure = objective.model.structure
        
        fit_qvals = np.linspace(0.006,0.26,1000)
        fit_spol = ani_ReflectModel(structure, scale=1,bkg=0,dq=0,Energy=energy,pol='s',name='spol')
        fit_ppol = ani_ReflectModel(structure, scale=1,bkg=0,dq=0,Energy=energy,pol='p',name='spol')
        
        data = objective.data
        res = objective.residuals()
        
        ##Split s and p pol
        swap_loc = np.argmax(np.abs(np.diff(data.data[0]))) ##Where does it swap from the maximum Q of spol to the minimum Q at ppol
        
        spol_q = data.data[0][:pol_swap_loc1+1]
        spol_data = data.data[1][:pol_swap_loc1+1]
        spol_u = data.data[2][:pol_swap_loc1+1]
        spol_res = res[:pol_swap_loc1+1]
        
        ppol_q = data.data[0][pol_swap_loc1+1:]
        ppol_data = data.data[1][pol_swap_loc1+1:]
        ppol_u = data.data[2][pol_swap_loc1+1:]
        ppol_res = res[pol_swap_loc1+1:]

        ##Plot stuff
        fig = plt.figure(constrained_layout=True,figsize=(12,9))
        grid = fig.add_gridspec(5, 5)
        refl_ax = fig.add_subplot(grid[1:,:])
        res_ax = fig.add_subplot(grid[0,:],sharex=refl_ax)
        
        res_spol = res_ax.scatter(fit_qvals, spol_res,color='#FE6100',marker='o',s=5,label='s-pol')
        res_ppol = res_ax.scatter(fit_qvals, ppol_res,color='#785EF0',marker='o',s=5,label='p-pol')
        
        res_ax.set_ylabel('Res [%]',fontsize=14)
        res_ax.tick_params(axis='both',labelsize=14)
        res_ax.set_ylim(-10,10)
        plt.setp(res_ax.get_xticklabels(), visible=False)


        spol_data_graph = refl_ax.errorbar(spol_qvals, spol_data, spol_u, color='#FE6100', marker='o', ms=8, ls="", markerfacecolor='w', markeredgewidth=2, zorder=10, label='s-pol')
        ppol_data_graph = refl_ax.errorbar(ppol_qvals, ppol_data, ppol_u, color='#785EF0', marker='o', ms=8, ls="", markerfacecolor='w', markeredgewidth=2, zorder=10, label='p-pol')
        fit_spol_graph = refl_ax.plot(fit_qvals, fit_spol.model(fit_qvals), color='0', lw=2,zorder=30, label='fit')
        fit_ppol_graph = refl_ax.plot(fit_qvals, fit_ppol.model(fit_qvals), color='0', lw=2,zorder=30)
        
        refl_ax.legend(fontsize=14)
        refl_ax.set_yscale('log')
        refl_ax.set_ylabel('Reflectivity',fontsize=14)
        refl_ax.set_xlabel('q [$\AA^{-1}$]',fontsize=14)
        refl_ax.tick_params(axis='both',labelsize=14)

        plt.savefig('Profile_Fits_' + name + '.png',dpi=300)
        plt.close()
        
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

        
    plt.savefig('corner_process.png',dpi=300)
    plt.close()
    
def plot_psoxr_corner(objective):
    
    important_params = ['thick', 'rough']    
    
    for objective in objective.objectives:
        labels = objective.parameters().names()




def save_integratedtime(objective, chain, name='auto_correlation_time'):
    import numpy as np
    list = objective.varying_parameters().names()
    
    labels = np.full(len(list) ,'',dtype=object)
    auto = np.full(len(list) ,0 ,dtype=Float)

    
    for i in range(len(list)):
        labels[i] = list[i].name
        auto[i] = integrated_time(chain,quiet=True)
    
    out = np.rollaxis(np.array([labels,auto]),1,0)
    np.savetxt(name + '.txt',out,fmt="%s, %f",delimiter=',')
    


    

def save_fitresults(objective, name='FitResults'):
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
            Param_names[i] = Param_list[i].name
            Param_vals[i] = Param_list[i].value
            Param_stderr[i] = Param_list[i].stderr
    
        for i in range(numConstraints):
            Constraint_names[i] = Constraint_list[i].name
            Constraint_vals[i] = Constraint_list[i].value
            Constraint_stderr[i] = Constraint_list[i].stderr
        
        Param_results = np.rollaxis(np.array([Param_names,Param_vals,Param_stderr]),1,0)
        Constraint_results = np.rollaxis(np.array([Constraint_names,Constraint_vals,Constraint_stderr]),1,0)
        np.savetxt('objective_'+ objective.name + '_vary_param.txt',Param_results,fmt="%s, %f, %f",delimiter=',')
        np.savetxt('objective_'+ objective.name + 'constraint_param.txt',Constraint_results,fmt="%s, %f, %f",delimiter=',')
    
    
    


def custom_corner(objective, vary_num=[], constraint_num=[], **kwds):
    
    var_pars = [objective.parameters.varying_parameters()[i] for i in vary_num]
    #cons_pars = [objective.parameters.constrained_parameters()[j] for j in constraint_num]
    chain = np.array([par.chain for par in var_pars])#, [con.chain for con in cons_pars]])
    chain = chain.reshape(len(chain),-1).T
    kwds["labels"] = labels
    kwds["quantiles"] = [0.16,0.5,0.84]
    return corner.corner(chain, **kwds)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
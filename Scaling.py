import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress, t
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import statsmodels.api as sm
import numpy as np

def scaling(x0,y0,name):
    x = np.log(x0)
    y = np.log(y0)
    res = linregress(x,y)
    y_hat = res.intercept + res.slope*x
    samis = y-y_hat
    # print("SAMIs = ",samis)
    print("Var = ",np.round(np.var(samis),2))
    x = np.log(x0)
    Y = np.log(y0)
    X = sm.add_constant(x)
    model = sm.OLS(Y,X)
    fit = model.fit(cov_type='HC1')
    # print(fit.summary())
        
    intercept, slope = fit.params


    x_0 =  np.sort(x0)[0]
    y_0 = np.exp(slope*np.log(x_0)+intercept)

    x_f = np.sort(x0)[-1]
    y_f = np.exp(slope*np.log(x_f)+intercept)
    y_null = np.exp(np.log(x_f)+intercept)

    betta = round(slope,3)
    R2 = str(round(fit.rsquared,2))

    beta_lowerbound, beta_upper = fit.conf_int().iloc[1]
    beta_lowerbound = np.round(beta_lowerbound,2)
    beta_upper = np.round(beta_upper,2)


    plt.rcParams.update({
    'font.size' : 20,
    "lines.linewidth": 2,
    "font.family":"arial",
    #"font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "mathtext.default": "rm",
    "mathtext.rm"  : "arial",
        })

    fig,(ax)  = plt.subplots(1, 1, sharey='row',figsize=(8, 8))
    ax.scatter(x0, y0, facecolors='#d4d4d4', edgecolors='k',s=50) ## #B36A6F,#508AB2,#D5BA82,#A1D0C7,#F0BB41,#a17db4,#ada579,#b3d6ad,#d6bbc1,#98A1B1,#F0E6E4,#d4d4d4
    ax.plot([x_0,x_f],[y_0,y_f], lw = 5, color = '#d4d4d4',label=r'$\beta = {}$'.format("{:.2f}".format(betta))+r'$ \, \in \,({}$'.format(beta_lowerbound) + r'$,{}]$'.format(beta_upper)+r', $\mathit{R}^2 = $' +r'${}$'.format('0.80'))
    ax.plot([x_0,x_f],[y_0,y_null],lw=3,color='#C52A20',linestyle='--',label=r'$\beta=1$')
    # ax.hlines(y = y_0, xmin=x_0,xmax=x_f,linestyle = '--', color="#C52A20") ## #934833
    ax.set_xlim([10**4,10**8])
    ax.set_ylim([10**1,10**7])

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Population")
    ax.set_ylabel(name)
    ax.legend( loc='upper left',frameon=False)

    x = np.log(x0)
    Y = np.log(y0/x0)
    X = sm.add_constant(x)
    model = sm.OLS(Y,X)
    fit2 = model.fit(cov_type='HC1')
    intercept2, slope2 = fit2.params
    print("intercept = ", np.round(intercept2,2))
    print("c = ", np.round(np.exp(intercept2),3))
    x_0 =  np.sort(x0)[0]
    y_0 = np.exp(slope2*np.log(x_0)+intercept2)
    x_f = np.sort(x0)[-1]
    y_f = np.exp(slope2*np.log(x_f)+intercept2)
    y_null = np.exp(np.log(x_f)+intercept2)

    # ax.set_yticks([0.1,1,10**1,10**2,10**3,10**4], minor=True)
    l, b, h, w = .53, .19, .25, .35
    ax2 = fig.add_axes([l, b, w, h])
    ax2.scatter(x0, y0/x0, facecolors='#d4d4d4', edgecolors='k',s=50) ## #B36A6F,#508AB2,#D5BA82,#A1D0C7,#F0BB41,#a17db4,#ada579,#b3d6ad,#d6bbc1,#98A1B1
    ax2.plot([x_0,x_f],[y_0,y_f], lw = 5, color = '#d4d4d4')
    ax2.hlines(y = y_0, xmin=x_0,xmax=x_f,linestyle = '--',lw=3, color="#C52A20") ## #934833
    ax2.set_xscale("log")
    # ax2.set_xticks([10**4,10**6,10**8])
    ax2.set_ylim([0,.3])
    ax2.set_ylabel('per-capita')

    # plt.savefig("{}.pdf".format(name),dpi=300, bbox_inches='tight')
    plt.show()
    return fit.conf_int(),samis,betta,float(R2)


def all_upper(my_list): 
    return list(map(lambda x: x.upper(), my_list))
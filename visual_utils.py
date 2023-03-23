#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns

from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import distance_transform_edt
import ase
from ase.io import cube
from ase.io import cif
import h5py
import pickle
import io
import json
import re
# from IPython.display import display, HTML
from scipy.spatial import Delaunay
import plotly
from plotly.graph_objs import graph_objs as go
import ipywidgets as widgets
# plotly.offline.init_notebook_mode(connected=True)
from plotly.offline import iplot
import gudhi, gudhi.hera, gudhi.wasserstein, persim
import mdtraj
import psutil
import argparse
# from main import get_args
import matplotlib as mpl
from captum.attr import Saliency

def saliency(coords):
    def forward_fn(coords):
        if self.args.which_model == 'fc_ae':
            z, x = self.model_block(coords)
            mu = z
        else:
            z, mu, logstd, x = self.model_block(coords)
        logprob = torch.distributions.Normal(0, 1).log_prob(mu).sum(dim=-1) #->(batch, )
        return logprob
    attrs = Saliency(forward_fn)

def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,
    alpha=1.,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    legend=True,
    show=False,
    ax=None,
    save="wass.png"
):
    """A helper function to plot persistence diagrams. 

    Parameters
    ----------
    diagrams: ndarray (n_pairs, 2) or list of diagrams
        A diagram or list of diagrams. If diagram is a list of diagrams, 
        then plot all on the same plot using different colors.
    plot_only: list of numeric
        If specified, an array of only the diagrams that should be plotted.
    title: string, default is None
        If title is defined, add it as title of the plot.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing 
        multiple persistence diagrams.
    labels: string or list of strings
        Legend labels for each diagram. 
        If none are specified, we use H_0, H_1, H_2,... by default.
    colormap: string, default is 'default'
        Any of matplotlib color palettes. 
        Some options are 'default', 'seaborn', 'sequential'. 
        See all available styles with

        .. code:: python

            import matplotlib as mpl
            print(mpl.styles.available)

    size: numeric, default is 20
        Pixel size of each point plotted.
    ax_color: any valid matplotlib color type. 
        See [https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html) for complete API.
    diagonal: bool, default is True
        Plot the diagonal x=y line.
    lifetime: bool, default is False. If True, diagonal is turned to False.
        Plot life time of each point instead of birth and death. 
        Essentially, visualize (x, y-x).
    legend: bool, default is True
        If true, show the legend.
    show: bool, default is False
        Call plt.show() after plotting. If you are using self.plot() as part 
        of a subplot, set show=False and call plt.show() only once at the end.
    """

    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = ["$H_{{{}}}$".format(i) for i , _ in enumerate(diagrams)]

    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label in zip(diagrams, labels):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, alpha=alpha, label=label, edgecolor="none")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
#         ax.legend(loc="lower right")
        ax.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

    if show is True:
        plt.show()
    if save is not None:
        plt.savefig("wass.png")
    
    
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 24
XLIM = [280, 330]
YLIM = [0, 0.12]
XTICKS = [280, 290, 300, 310, 320, 330]
YTICKS = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]

def plot_total_temps(filename: str):
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    XLIM = [280, 330]
    YLIM = [0, 0.08]
    XTICKS = [280, 290, 300, 310, 320, 330]
    YTICKS = [0, 0.02, 0.04, 0.06, 0.08]
    
    assert os.path.splitext(filename)[1] == ".npz", "File name extension is wrong..."
    data = np.load(filename)
    keys = list(data)
    BINS = 100
    
    fig, ax = plt.subplots() 
    ax.hist(data["pred"], bins=BINS, density=True, alpha=0.2, color='b') #npz has pred; pickle has predictions
    sns.kdeplot(data=data["pred"].reshape(-1, ), ax=ax, color='k', fill=False, common_norm=False, alpha=1, linewidth=2)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("Effective Temperatures ($\mathregular{T_E}$)")
    ax.set_ylabel("PDF")
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)

    ax.set_title("Effective Temperature Distribution")
#     ax.set_ylim(280, 330)
    fig.savefig(os.path.splitext(filename)[0] + ".png")

#     with Parallel(n_jobs=psutil.cpu_count(), backend='multiprocessing') as parallel:
#         results = parallel(delayed(calc_2d_filters)(toks, pains_smarts) for count, toks in enumerate(data)) #List[List]

def plot_one_temp(filename: str):
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    XLIM = [280, 330]
    YLIM = [0, 0.16]
    XTICKS = [280, 290, 300, 310, 320, 330]
    YTICKS = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    
    assert os.path.splitext(filename)[1] == ".pickle", "File name extension is wrong..."
    assert "Predicted" in os.path.basename(filename), "File name prefix is wrong..."

    f = open(filename, "rb")
    data = pickle.load(f)
    keys = data.keys()
    BINS = 100
    
    fig, ax = plt.subplots() 
    if "DPPC" in os.path.basename(filename):
        ax.hist(data["predictions"].detach().cpu().numpy(), bins=BINS, density=True, alpha=0.2, color='r') #npz has pred; pickle has predictions
        YLIM = [0, 0.08]
        YTICKS = [0, 0.02, 0.04, 0.06, 0.08]
    elif "B2GP1" == os.path.basename(filename).split("_")[0]:
        ax.hist(data["predictions"].detach().cpu().numpy(), bins=BINS, density=True, alpha=0.2, color='g') #npz has pred; pickle has predictions
        YLIM = [0, 0.16]
        YTICKS = np.linspace(0, 0.16, 9).tolist()
    else:
        ax.hist(data["predictions"].detach().cpu().numpy(), bins=BINS, density=True, alpha=0.2, color='g') #npz has pred; pickle has predictions
        YLIM = [0, 0.24]
        YTICKS = np.linspace(0, 0.24, 13).tolist()
    sns.kdeplot(data=data["predictions"].detach().cpu().numpy().reshape(-1, ), ax=ax, color='k', fill=False, common_norm=False, alpha=1, linewidth=2)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("Effective Temperatures ($\mathregular{T_E}$)")
    ax.set_ylabel("PDF")
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)
    
#     ax.set_title(f"Effective Temperature Distribution - { int(os.path.basename(os.path.splitext(filename)[0]).split('_')[-1]) } K")
    ax.set_title(f"Effective Temperature Distribution")

#     ax.set_ylim(280, 330)
    fig.savefig(os.path.splitext(filename)[0] + ".png")

def plot_one_temp_parallel(args: argparse.ArgumentParser):
    ROOT_DIR = args.save_dir
    filenames_ = os.listdir(ROOT_DIR)
    filenames = list(filter(lambda inp: ("Predicted" in os.path.basename(inp) and os.path.splitext(inp)[1] == ".pickle"), filenames_ ))
#     filenames_bools = list(map(lambda inp: ("Predicted" in os.path.basename(inp) and os.path.splitext(inp)[1] == ".png"), filenames_ )) #List[bool]
    filenames = list(map(lambda inp: os.path.join(ROOT_DIR, inp), filenames ))
#     filenames = np.array(filenames)[~np.array(filenames_bools)].tolist() #only without pngs
    print(filenames)
    
    from time import perf_counter
    
    if args.multiprocessing_backend == "multiprocessing":
        t_start = perf_counter()
        from multiprocessing import Pool
        with Pool(processes=psutil.cpu_count()) as pool:
            results = pool.map(plot_one_temp, filenames)
        t_stop = perf_counter()
        print(f"Multiprocessing took {t_stop - t_start} seconds...")
        
    if args.multiprocessing_backend == "dask":
        t_start = perf_counter()
        import dask
        results = [dask.delayed(plot_one_temp(filename)) for filename in filenames] #analogous to [func.remote(args) for args in args_list]
        results = dask.compute(results)
        t_stop = perf_counter()
        print(f"Dask took {t_stop - t_start} seconds...")
    
    if args.multiprocessing_backend == "joblib":
        t_start = perf_counter()
        from joblib import Parallel, delayed
        with Parallel(n_jobs=psutil.cpu_count(), backend='loky') as parallel:
            results = parallel(delayed(plot_one_temp)(filename) for idx, filename in enumerate(filenames)) #List[None]
        t_stop = perf_counter()
        print(f"Joblib took {t_stop - t_start} seconds...")
    
    if args.multiprocessing_backend == "ray":
        t_start = perf_counter()
        import ray.util.multiprocessing as mp
        pool = mp.Pool(processes=psutil.cpu_count())
        results = pool.map_async(plot_one_temp, filenames)
        t_stop = perf_counter()
        print(f"Ray took {t_stop - t_start} seconds...")
    
def genAlphaSlider(dat,initial=1,step=1,maximum=10,titlePrefix=""): #assume 3D for now
    ac = gudhi.AlphaComplex(dat)
    st = ac.create_simplex_tree()
    skel=list(st.get_skeleton(2))
    skel.sort(key=lambda s: s[1])
    points = np.array([ac.get_point(i) for i in range(st.num_vertices())])
    #lims=[[np.floor(np.min(dat[:,i])),np.ceil(np.max(dat[:,i]))] for i in range(3)]
    alpha = widgets.FloatSlider(
        value = initial,
        min = 0.0,
        max = maximum,
        step = step,
        description = 'Alpha:',
        readout_format = '.4f'
    )



    b1s=np.array([s[0] for s in skel if len(s[0]) == 2 and s[1] <= alpha.value])
    triangles = np.array([s[0] for s in skel if len(s[0]) == 3 and s[1] <= alpha.value])


    pts=go.Scatter3d(
        x = points[:, 0],
        y = points[:, 1],
        z = points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color="cornflowerblue",                # set color to an array/list of desired values
            #colorscale='Viridis',   # choose a colorscale
            opacity=.9
        ),
        name='H0'

    )

    sfig=[pts]

    linepts={0:[],1:[],2:[]}
    for i in b1s:
        linepts[0].append(points[i[0],0])
        linepts[1].append(points[i[0],1])
        linepts[2].append(points[i[0],2])
        linepts[0].append(points[i[1],0])
        linepts[1].append(points[i[1],1])
        linepts[2].append(points[i[1],2])

        linepts[0].append(None)
        linepts[1].append(None)
        linepts[2].append(None)

    if len(linepts[0])>0:
        lins=go.Scatter3d(
            x=linepts[0],
            y=linepts[1],
            z=linepts[2],
            mode='lines',
            name='H1',
            marker=dict(
                size=3,
                color="#d55e00",                # set color to an array/list of desired values
                #colorscale='Viridis',   # choose a colorscale
                opacity=.9
            )
        )
        sfig.append(lins)
        if len(triangles)>0:
            mesh = go.Mesh3d(
                x = points[:, 0],
                y = points[:, 1],
                z = points[:, 2],
                i = triangles[:, 0],
                j = triangles[:, 1],
                k = triangles[:, 2],
                color="#009e73",
                opacity=.75,
                name='H2'
            )


            sfig.append(mesh)
    fig=go.Figure(sfig)
    fig.update_layout(width=800,height=800)
    #fig.show()




    def view_SC(alpha):
        if alpha==0:
            fig=go.Figure(sfig[0])
            fig.show()
        else:
            b1s=np.array([s[0] for s in skel if len(s[0]) == 2 and s[1] <= alpha])

            linepts={0:[],1:[],2:[]}
            for i in b1s:
                linepts[0].append(points[i[0],0])
                linepts[1].append(points[i[0],1])
                linepts[2].append(points[i[0],2])
                linepts[0].append(points[i[1],0])
                linepts[1].append(points[i[1],1])
                linepts[2].append(points[i[1],2])

                linepts[0].append(None)
                linepts[1].append(None)
                linepts[2].append(None)

            if len(linepts[0])>0:
                lins=go.Scatter3d(
                    x=linepts[0],
                    y=linepts[1],
                    z=linepts[2],
                    mode='lines',
                    name='H1',
                    marker=dict(
                        size=3,
                        color="#d55e00",                # set color to an array/list of desired values
                        #colorscale='Viridis',   # choose a colorscale
                        opacity=.85
                    )
                )
                if len(sfig)>1:
                    sfig[1]=lins
                else:
                    sfig.append(lins)
                triangles = np.array([s[0] for s in skel if len(s[0]) == 3 and s[1] <= alpha])
                if len(triangles)>0:
                    mesh = go.Mesh3d(
                        x = points[:, 0],
                        y = points[:, 1],
                        z = points[:, 2],
                        i = triangles[:, 0],
                        j = triangles[:, 1],
                        k = triangles[:, 2],
                        color="#009e73",
                        opacity=.5,
                        name='H2'
                    )

                    if len(sfig)>2:
                        sfig[2]=mesh
                    else:
                        sfig.append(mesh)


            fig=go.Figure(data=sfig,layout=go.Layout(width=800,height=800,
                                                     title=f"{titlePrefix}:\nSimplicial complex with radius <= {round(float(alpha),5)}",
                                                    ))

            #fig.show()
            iplot(fig)


    widgets.interact(view_SC, alpha = alpha);
    return st
  
if __name__ == "__main__":
    from main import get_args
    args = get_args()
    plot_total_temps(os.path.join(args.save_dir, "convnext_model_indiv_all_temps.npz"))
    plot_one_temp_parallel(args)
#     pdb = args.pdb
#     data = mdtraj.load(pdb, top=pdb)
#     genAlphaSlider(data.xyz[0], initial=1, step=1, maximum=10, titlePrefix="")

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
from joblib import Parallel, delayed
import psutil
from main import get_args

def plot_total_temps(filename: str):
    assert os.path.splitext(filename)[1] == ".npz", "File name extension is wrong..."
    data = np.load(filename)
    keys = list(data)
    BINS = 100
    
    fig, ax = plt.subplots() 
    ax.hist(data["pred"], bins=BINS, density=True, alpha=0.2) #npz has pred; pickle has predictions
    sns.kdeplot(data=data["pred"].reshape(-1, ), ax=ax, color='k', mp)
    ax.set_xlim(280, 330)
    ax.set_ylim(0, 0.08)
    ax.set_xlabel("Temperatures")
    ax.set_ylabel("PDF")
    ax.set_xticks([280, 290, 300, 310, 320, 330])
    
    ax.set_title("All Temperatures")
#     ax.set_ylim(280, 330)
    fig.savefig(os.path.splitext(filename)[0] + ".png")

#     with Parallel(n_jobs=psutil.cpu_count(), backend='multiprocessing') as parallel:
#         results = parallel(delayed(calc_2d_filters)(toks, pains_smarts) for count, toks in enumerate(data)) #List[List]

def plot_one_temp(filename: str):
    assert os.path.splitext(filename)[1] == ".pickle", "File name extension is wrong..."
    assert os.path.basename(filename).startswith("Predicted"), "File name prefix is wrong..."

    f = open(filename, "rb")
    data = pickle.load(f)
    keys = data.keys()
    BINS = 100
    
    fig, ax = plt.subplots() 
    ax.hist(data["predictions"].detach().cpu().numpy(), bins=BINS, density=True, alpha=0.2) #npz has pred; pickle has predictions
    sns.kdeplot(data=data["predictions"].detach().cpu().numpy().reshape(-1, ), ax=ax, color='k')
    ax.set_xlim(280, 330)
    ax.set_ylim(0, 0.08)
    ax.set_xlabel("Temperatures")
    ax.set_ylabel("PDF")
    ax.set_xticks([280, 290, 300, 310, 320, 330])
    
    ax.set_title(f"Indiv Lipid Temp Predictions at { int(os.path.basename(os.path.splitext(filename)[0]).split('_')[-1]) } Kelvin")
#     ax.set_ylim(280, 330)
    fig.savefig(os.path.splitext(filename)[0] + ".png")

def plot_one_temp_parallel(args: argparse.ArgumentParser):
    ROOT_DIR = args.save_dir
    filenames = os.listdir(ROOT_DIR)
    filenames = list(filter(lambda inp: ("Predicted" in os.path.basename(inp) and os.path.splitext(inp)[1] == ".pickle"), filenames ))
    filenames_bools = list(map(lambda inp: os.path.splitext(inp)[1] == ".png", filenames )) #List[bool]
    filenames = list(map(lambda inp: os.path.join(ROOT_DIR, inp), filenames ))
    filenames = np.array(filenames)[~np.array(filenames_bools)].tolist() #only without pngs
    print(filenames)
    
    with Parallel(n_jobs=psutil.cpu_count(), backend='multiprocessing') as parallel:
        results = parallel(delayed(plot_one_temp)(filename) for idx, filename in enumerate(filenames)) #List[None]
    
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
    plot_one_temp_parallel(args)
#     pdb = args.pdb
#     data = mdtraj.load(pdb, top=pdb)
#     genAlphaSlider(data.xyz[0], initial=1, step=1, maximum=10, titlePrefix="")

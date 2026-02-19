from ase import io
from ase.geometry import get_distances
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append("/projects/bcqo/shubhanggoswami/paul_ricky_codes/solid_hydrogen")
sys.path.append("/projects/bcqo/shubhanggoswami/paul_ricky_codes/harvest_qmcpack")
from qharv_db import grsk, ipi_md
import argparse
import pandas as pd
import time
import glob

def reblock(trace, block_size, min_nblock=4, with_sigma=False):
    """ block scalar trace to remove auocorrelation;
    see usage example in reblock_scalar_df
    Args:
        trace (np.array): a trace of scalars, may have multiple columns
        !!!! assuming leading dimension is the number of current blocks.
        block_size (int): size of block in units of current block.
        min_nblock (int,optional): minimum number of blocks needed for
        meaningful statistics, default is 4.
    Returns:
        np.array: re-blocked trace.
    """
    nblock= len(trace)//block_size
    nkeep = nblock*block_size
    if (nblock < min_nblock):
        raise RuntimeError('only %d blocks left after reblock' % nblock)
    # end if
    blocked_trace = trace[:nkeep].reshape(nblock, block_size, *trace.shape[1:])
    ret = np.mean(blocked_trace, axis=1)
    if with_sigma:
        ret = (ret, np.std(blocked_trace, ddof=1, axis=1)/np.sqrt(block_size))
    return ret

def find_dimers(rij, rmax=np.inf, rmin=0, sort_id=False):
    """
    from Paul harvest_qmcpack axes_pos.py 
    find all dimers within a separtion of (rmin, rmax)
    Args:
        rij  (np.array): distance table
        rmax (float, optional): maximum dimer separation, default np.inf
        rmin (float, optional): minimum dimer separation, default 0
        sort_id (bool, optional): sort pair by first atom id, default false
    Return:
        np.array: unique pairs, a list of (int, int) particle id pairs
    """
    natom, natom1 = rij.shape
    assert natom1 == natom
    found = np.zeros(natom, dtype=bool)
    pairs = []
    # loop through pair distances from small to large
    idx = np.triu_indices(natom, 1)
    dists = rij[idx]
    ij = np.array(idx).T
    for idist in np.argsort(dists):
        i, j = ij[idist]  # pair indices
        if found[i] or found[j]:
            continue
        rb = dists[idist]  # bond length
        if (rb < rmin) or (rb > rmax):
            continue
        pair = (i, j) if i < j else (j, i)
        pairs.append(pair)
        found[i] = True
        found[j] = True
        if np.all(found):
            break
    pa = np.array(pairs)
    if sort_id:  # sort pairs by first atom id
        i1 = np.argsort(pa[:, 0])
        sorted_pairs = pa[i1]
    else:
        sorted_pairs = pa
    return sorted_pairs

def get_pos_and_axes(path_to_dump_file,format='lammps-dump-text'):
    posl= []
    axesl=[]
    # Read atoms
    traj = io.read(path_to_dump_file,format=format,index=':')
    # Get atoms coordinates
    for i, atoms in enumerate(traj):
        # axes
        axes  = atoms.get_cell()
        axesl.append(axes)
        # positions
        pos   = atoms.get_positions()
        posl.append(pos)
    return(axesl,posl)



def calc_sofk_npt_time(bin_edges, axesl, posl, nsh=8):
    kmin = bin_edges.min()
    kmax = bin_edges.max()
    dk = bin_edges[1]-bin_edges[0]
    dk1 = bin_edges[2]-bin_edges[1]
    if not np.isclose(dk, dk1):
        msg = 'not a linear grid'
        raise RuntimeError(msg)
    nk = len(bin_edges) - 1
    skl = []
    for axes, pos in zip(axesl, posl):
        kvecs = grsk.legal_kvecs(axes, nsh)
        kmags = np.linalg.norm(kvecs, axis=-1)
        if kmags.max() < kmax:
            msg = 'increase nsh=%d for kmax=%f' % (nsh, kmax)
            raise RuntimeError(msg)
        
        # #choose kmags that lie within 10% of pure hcp crystal
        # kmax = 1.1*3.5705
        # kmin = 0.9*3.5705

        vec_idx = (kmags <= kmax) & (kmags >= kmin)
        sel_kvecs = kvecs[vec_idx]
        #sel_kvecs = kvecs
        sk1 = grsk.Sk(sel_kvecs, pos)
        k_w_sofk = np.column_stack((sel_kvecs.tolist(),sk1))
        skl.append(k_w_sofk[k_w_sofk[:,3] > 10])
        
#   uskm, uske = yl_ysql(skl)
    return np.vstack(skl)

def get_rs(axesl,N):
    vol = np.array([np.linalg.det(ax) for ax in axesl])
    V = reblock(vol,10)
    coeff=(4/3)*np.pi
    rs=(V/(N*coeff))**(1/3)
    return np.mean(rs)

def lcount(keyword, fname):
    with open(fname, 'r') as fin:
        return sum([1 for line in fin if keyword in line])

def get_mic_distances_from_vecs(vec1,vec2,cell,pbc):
    assert len(vec1)==len(vec2)
    v = np.zeros(np.shape(vec1))
    v_norm = np.zeros(len(vec1))
    for idx,row in enumerate(vec1):
        v[idx,:],v_norm[idx] = get_distances(row,vec2[idx,:],cell=cell,pbc=pbc)

    return v,v_norm

def get_dimer_axes_posl(data_path,case):
    #num_of_trajs = lcount("ITEM: TIMESTEP",data_path+"dump.merged.atom")
    #set_of_all_dimerpairs = {}
    start=time.time()
    axl = []
    posl = []
    if "ASE" in case:
        fname = glob.glob(data_path,"*.merged.traj")[0]
        trajectories = io.read(data_path+fname,index=':')
    else:
        trajectories = io.read(data_path+"dump.merged.atom",format='lammps-dump-text',index=':')
    R_norm = trajectories[0].get_all_distances(mic=True,vector=False)
    num_atoms = trajectories[0].get_global_number_of_atoms()
    num_dimers = int(num_atoms/2)
    dimer_pairs = find_dimers(R_norm,1.5,sort_id=True)
    for t,traj in enumerate(trajectories):
        axl.append(traj.get_cell())
        rij = traj.get_positions()
        atoms_H1 = rij[dimer_pairs[:,0],:]
        atoms_H2 = rij[dimer_pairs[:,1],:]
        dimer_rijs, dimer_norms = get_mic_distances_from_vecs(atoms_H1,atoms_H2,cell=traj.cell,pbc=traj.pbc)
        dimer_rij_pos = atoms_H1+0.5*dimer_rijs
        posl.append(dimer_rij_pos)
    end=time.time()
    print(f"dimer axes and positions calculated in {end-start} seconds")
    return axl,posl
    

def main():
    parser = argparse.ArgumentParser(description='input P-T to create a two-phase starting configuration from prev results')
    parser.add_argument('-p','--press',dest='pressure',type=int,help='pressure',required=True)
    parser.add_argument('-t','--temp',dest='temp',type=int,help='temperature',required=True)
    parser.add_argument('-c','--case',dest='case',type=str,help="NPT or NVT or something else",default="NPT")
    parser.add_argument('-f','--phase',dest='phase',type=str,help="solid or liquid phase",default="liquid")
    parser.add_argument('-n','--natom',dest='natom',type=int,help='number of atoms',required=True)
    parser.add_argument('--dimer',dest="dimer",action='store_true',help="Tag to turn on bragg peak calculation for dimer")
    args= parser.parse_args()
    pgpa= args.pressure
    tkel = args.temp
    case = args.case
    phase = args.phase

    N_atoms = args.natom
    data_path=f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N_atoms}atoms/p{pgpa}/{phase}/{case}/{tkel}/analysis/"
    
    pngsavepath = data_path+f"P{pgpa}T{tkel}bragg_peaks_over_time_unitless"
    data_save_path=data_path+f"P{pgpa}T{tkel}bragg_peaks_over_time_nearhcp"
    if args.dimer:
        axesl,posl=get_dimer_axes_posl(data_path,case)
        pngsavepath+="_dimer"
        data_save_path+="_dimer"
    else:
        if "ASE" in case:
            axesl,posl = get_pos_and_axes(data_path+f'P{pgpa}T{tkel}.merged.traj',format="traj")
        else:
            axesl,posl = get_pos_and_axes(data_path+f'dump.merged.atom')
    

    L_box= max([max(np.diag(ax)) for ax in axesl])
    rs = get_rs(axesl,N_atoms)
    bin_edges = np.arange(2/rs,5/rs,2*np.pi/L_box)
    nbins = len(bin_edges)

    sk_t = calc_sofk_npt_time(bin_edges,axesl,posl,nbins)
    save_sk = pd.DataFrame(sk_t)
    save_sk.columns = ["kx","ky","kz","sk"]
    save_sk.to_csv(data_save_path+".csv",index=False)
        

    timesteps,ncol =np.shape(sk_t)
    kmags = np.linalg.norm(sk_t[:,:3],axis=-1)
    kmags_max1 = 3.5/rs
    kmags_min1 = 3/rs

    kmags_max2 = 4/rs
    kmags_min2 = 3.5/rs

    kmags_max3 = 4.5/rs
    kmags_min3 = 4/rs

    vec_idx0 = (kmags <= 3/rs)
    vec_idx1 = (kmags < kmags_max1) & (kmags >= kmags_min1)
    vec_idx2 = (kmags < kmags_max2) & (kmags >= kmags_min2)
    vec_idx3 = (kmags < kmags_max3) & (kmags >= kmags_min3)
    vec_idx4 = (kmags >= 4.5/rs)

    plt.scatter(np.arange(timesteps)[vec_idx0],sk_t[:,3][vec_idx0],label="|k|r_s<=3",marker='s',alpha=0.7,s=5)
    plt.scatter(np.arange(timesteps)[vec_idx1],sk_t[:,3][vec_idx1],label="3<|k|r_s<3.5",marker='+',alpha=0.7,s=5)
    plt.scatter(np.arange(timesteps)[vec_idx2],sk_t[:,3][vec_idx2],label="3.5<|k|r_s<4",marker='D',alpha=0.7,s=5)
    plt.scatter(np.arange(timesteps)[vec_idx3],sk_t[:,3][vec_idx3],label="4<|k|r_s<4.5",marker="v",alpha=0.7,s=5)
    plt.scatter(np.arange(timesteps)[vec_idx4],sk_t[:,3][vec_idx4],label="|k|r_s>4.5",marker="*",alpha=0.7,s=5)
    plt.title(f"P={pgpa} GPa, T={tkel} K {phase} phase")
    plt.ylabel("S(k)")
    plt.legend()
    plt.savefig(pngsavepath+".png")

if __name__ == '__main__':
    main()

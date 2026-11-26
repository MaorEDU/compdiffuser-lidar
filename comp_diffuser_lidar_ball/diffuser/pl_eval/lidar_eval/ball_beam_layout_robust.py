"""Is the 8-beam win an artifact of the per-config tau normalisation?
Re-score every layout against (a) its own tau, (b) a COMMON tau, and (c) a
scale-free criterion that needs no tau at all: is the nearest FAR look-alike
closer than the nearest LEGAL-neighbour pose? That last one is a pure ranking
question -- immune to any per-config scaling."""
import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location('lidar_grid', 'diffuser/datasets/lidar/lidar_grid.py')
lg = importlib.util.module_from_spec(spec); spec.loader.exec_module(lg)
ENV='pointmaze-giant-stitch-v0'
def build(nb, mr):
    g = lg.build_grid_for_env(ENV, lidar_config=dict(n_beams=nb, max_range=mr),
                              xy_step=0.5, n_theta=1, verbose=False)
    return g.free_xy.astype(np.float64), g.free_scans.astype(np.float64)
def pdist2(A):
    sq=(A*A).sum(1); d2=sq[:,None]+sq[None,:]-2.0*(A@A.T); np.maximum(d2,0.0,out=d2); return d2
print(f"{'cfg':10}{'tau':>8}{'conf(own tau)':>15}{'conf(common tau)':>18}{'SCALE-FREE: far<legal':>24}{'exact dupes':>13}")
rows={}
for nb,mr in [(4,12.),(8,12.),(16,12.),(30,12.),(60,12.),(8,24.),(30,24.)]:
    xy,S=build(nb,mr); M=len(S)
    d2xy=pdist2(xy); rms=np.sqrt(pdist2(S))/np.sqrt(nb)
    neigh=(d2xy>1e-9)&(d2xy<=0.51**2)
    tau=float(np.median([np.median(rms[i,neigh[i]]) for i in range(M)]))
    far=d2xy>=8.0**2
    alias=np.where(far,rms,np.inf).min(1)
    # scale-free: nearest far look-alike vs nearest legal neighbour, same units
    near_legal=np.array([rms[i,neigh[i]].min() if neigh[i].any() else np.inf for i in range(M)])
    rows[(nb,mr)]=dict(tau=tau,alias=alias,near_legal=near_legal)
COMMON=np.median([r['tau'] for r in rows.values()])
for (nb,mr),r in rows.items():
    a=r['alias']
    print(f"{f'{nb}b/mr{mr:g}':10}{r['tau']:>8.3f}{100*np.mean(a<r['tau']):>14.1f}%"
          f"{100*np.mean(a<COMMON):>17.1f}%{100*np.mean(a<r['near_legal']):>23.1f}%"
          f"{100*np.mean(a<=1e-9):>12.1f}%")
print(f"\ncommon tau = {COMMON:.3f} (median over configs)")
print("SCALE-FREE column = frac of poses whose nearest >=8wu look-alike is CLOSER in")
print("scan space than its own nearest 0.5wu neighbour. No tau, no threshold, no units.")

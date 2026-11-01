"""Run using assignment2/venv/bin/python -B run_experiment.py. Only uses snapshot code.
All 34 reduced problems use identical calibration and forward-validation data.
"""
from pathlib import Path
import os,sys,json,time,csv,contextlib,io
import numpy as np
from scipy.optimize import least_squares

OUT=Path(__file__).resolve().parent
ROOT=OUT/'source_snapshot'
A2=ROOT/'assignments/assignment2'
sys.path.insert(0,str(A2))
os.chdir(A2/'notebooks')
nb=json.loads((A2/'notebooks/assignment2.ipynb').read_text())
np.random.seed(20261025)
g={}
with contextlib.redirect_stdout(io.StringIO()):
    for i in [1,3,5,7,15,17]:
        exec(compile(''.join(nb['cells'][i]['source']),f'notebook_cell_{i}','exec'),g)
model=g['kinematics'].ImprovedKinematics(g['rob'],g['F_X_nom'],joint_offsets=g['home_offsets'])
data=g['data']
names=['base_alpha_x','base_alpha_y','base_alpha_z','base_rho_x','base_rho_y','base_rho_z',
       'd2_phi_x','d2_phi_z','d3_phi_x','d3_phi_y','n4_phi_y','n4_phi_z',
       'n5_phi_x','n5_phi_z','n6_phi_x','n6_phi_y',
       'tool_alpha_x','tool_alpha_y','tool_alpha_z','tool_rho_x','tool_rho_y','tool_rho_z']
names += [f'joint{k}_c{j}' for k in range(1,7) for j in range(1,3)]
units=['rad']*3+['mm']*3+['rad']*13+['mm']*3+['mm']*6+['rad']*6

# Batched evaluation of the identical model avoids rebuilding constant axes/base/tool
# for every calibration sample. Original residuals are retained for cross-checks.
from uncertainty_networks.se3 import exp_so3, exp_se3
from data_types.nominal_types import Frame
T=np.array(data.thetas)
OBS_R=np.array([f.R.matrix for f in data.frames])
OBS_P=np.array([f.p.vec.ravel() for f in data.frames])
N=len(T)
assert np.allclose(model.F_CW.R.matrix,np.eye(3))

def fast_residuals(p):
    scales=np.maximum(abs(model.theta_min),abs(model.theta_max))
    coeff=p[22:].reshape(6,model.n_poly)
    q=T+model.joint_offsets
    for j in range(model.n_poly):q=q+(T/scales)**(j+1)*coeff[:,j]
    axes=np.array([a.vec.ravel() for a in model.AXES])
    for k in range(1,6):
        phi=np.zeros(3);phi[axes[k]==0]=p[6+2*(k-1):8+2*(k-1)]
        axes[k]=exp_so3(phi)@axes[k]
    rw=np.broadcast_to(np.eye(3),(N,3,3)).copy()
    for k in range(3,6):
        x,y,z=axes[k]
        K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        rk=np.eye(3)+np.sin(q[:,k])[:,None,None]*K+(1-np.cos(q[:,k]))[:,None,None]*(K@K)
        rw=rw@rk
    tool=model.F_WE*Frame(exp_se3(p[16:22]));base=model.base_frame(p)
    pos=q[:,:3]@axes[:3]+model.F_CW.p.vec.ravel()+np.einsum('nij,j->ni',rw,tool.p.vec.ravel())
    pp=pos@base.R.matrix.T+base.p.vec.ravel()
    rr=base.R.matrix[None,:,:]@rw@tool.R.matrix[None,:,:]
    er=OBS_R.transpose(0,2,1)@rr
    ep=np.einsum('nji,nj->ni',OBS_R,pp-OBS_P)
    theta=np.arccos(np.clip((np.trace(er,axis1=1,axis2=2)-1)*.5,-1,1))
    factor=np.full(N,.5);mask=theta>=1e-12
    factor[mask]=theta[mask]/(2*np.sin(theta[mask]))
    vee=np.column_stack([er[:,2,1]-er[:,1,2],er[:,0,2]-er[:,2,0],er[:,1,0]-er[:,0,1]])
    alpha=factor[:,None]*vee
    return np.column_stack([alpha/1e-3,ep/.2]).ravel()

def jacobian(p):
    r=model.residuals(p,data)
    J=np.empty((len(r),len(p)))
    for k in range(len(p)):
        step=1e-6*max(1.,abs(p[k]));above=p.copy();below=p.copy()
        above[k]+=step;below[k]-=step
        J[:,k]=(model.residuals(above,data)-model.residuals(below,data))/(2*step)
    return J

def diagnostics(J):
    sv=np.linalg.svd(J,compute_uv=False)
    tol=sv[0]*1e-6
    cn=np.linalg.norm(J,axis=0)
    ns=np.linalg.svd(J/np.where(cn>0,cn,1),compute_uv=False)
    return dict(rank=int(np.sum(sv>tol)),sigma_max=float(sv[0]),sigma_min=float(sv[-1]),
                condition=float(sv[0]/sv[-1]) if sv[-1]>0 else float('inf'),
                rank_1e5=int(np.sum(sv>sv[0]*1e-5)),rank_1e7=int(np.sum(sv>sv[0]*1e-7)),
                normalized_rank=int(np.sum(ns>ns[0]*1e-6)))

start=time.perf_counter();fit=model.fit(data);p0=model.params.copy();r0=model.residuals(p0,data)
rng_check=np.random.default_rng(12345)
check_errors=[]
for pc in [p0]+[p0+rng_check.normal(0,1e-3,34) for _ in range(10)]:
    expected=model.residuals(pc,data);actual=fast_residuals(pc)
    check_errors.append(float(np.max(abs(expected-actual))))
    np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=1e-9)
print('BATCH_EQUIVALENCE_MAX_ABS',max(check_errors),flush=True)
J0=jacobian(p0);U,s0,Vt=np.linalg.svd(J0,full_matrices=False);v=Vt[-1]
# Deterministic sign for presentation only.
if v[np.argmax(abs(v))]<0:v=-v
np.savez(OUT/'frozen_data.npz',thetas=np.array(data.thetas),
         observed_R=np.array([f.R.matrix for f in data.frames]),observed_p=np.array([f.p.vec.ravel() for f in data.frames]),
         fitted_parameters=p0,jacobian=J0,weak_vector=v,singular_values=s0,
         F_X_init_R=model.F_X_init.R.matrix,F_X_init_p=model.F_X_init.p.vec.ravel(),home_offsets=model.joint_offsets)
base=dict(fit=fit,diagnostics=diagnostics(J0),cost=float(.5*r0@r0),seed=1,noise_seed=20261025,n_samples=len(data),names=names,units=units,parameters=p0.tolist(),weak_vector=v.tolist(),singular_values=s0.tolist())
(OUT/'baseline.json').write_text(json.dumps(base,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
# Fixed evaluation input and oracle targets; never passed to fitting.
levels=[(100.,300.,500.),(-200.,0.,200.),(100.,300.,500.),(-1.3,0.,1.3),(-1.3,0.,1.3),(-2.,0.,2.)]
import itertools
poses=[np.array(t) for t in itertools.product(*levels)]
poses=[t for t in poses if g['trk'].is_visible(g['mbee_markers_world'](t))]
Ftr=g['trk'].nominal_pose()
truth=[Ftr.inv()*g['rob'].actual_ee_pose(t) for t in poses]
def evaluate(p):
    errors=[g['metrics'].frame_error(F,model.predict_in_tracker(t,p)) for t,F in zip(poses,truth)]
    return g['metrics'].summarize_errors(errors)
base['forward']=evaluate(p0)
(OUT/'baseline.json').write_text(json.dumps(base,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
print('BASELINE',json.dumps({'rank':base['diagnostics']['rank'],'cost':base['cost'],'forward':base['forward']}),flush=True)
rows=[]
for k,name in enumerate(names):
    active=np.arange(34)!=k
    same=diagnostics(J0[:,active])
    # Fix parameter at zero, retain baseline estimates for all other starting values.
    # This is a genuine new fit, not merely removing a column from the original J.
    def expand(x):
        p=np.zeros(34);p[active]=x;return p
    t=time.perf_counter()
    result=least_squares(lambda x:fast_residuals(expand(x)),p0[active],
                         x_scale='jac',max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    p=expand(result.x)
    np.testing.assert_allclose(fast_residuals(p),model.residuals(p,data),atol=2e-8,rtol=1e-9)
    jr=jacobian(p);post=diagnostics(jr[:,active]);full=diagnostics(jr)
    forward=evaluate(p)
    row=dict(index0=k,index1=k+1,parameter=name,unit=units[k],baseline_value=float(p0[k]),
             weak_component=float(v[k]),weak_abs=float(abs(v[k])),
             delete_rank=same['rank'],delete_sigma_min=same['sigma_min'],delete_condition=same['condition'],
             delete_rank_common_tol=int(np.sum(np.linalg.svd(J0[:,active],compute_uv=False)>s0[0]*1e-6)),
             delete_rank_1e5=same['rank_1e5'],delete_rank_1e7=same['rank_1e7'],delete_normalized_rank=same['normalized_rank'],
             fixed_value=0.,refit_success=bool(result.success),refit_nfev=int(result.nfev),
             refit_cost=float(result.cost),cost_change=float(result.cost-base['cost']),
             refit_rank=post['rank'],refit_full_rank=full['rank'],refit_sigma_min=post['sigma_min'],
             refit_condition=post['condition'],refit_normalized_rank=post['normalized_rank'],
             rms_pos=forward['rms_pos'],max_pos=forward['max_pos'],rms_angle=forward['rms_angle'],max_angle=forward['max_angle'],
             forward_pass=bool(forward['max_pos']<=.2 and forward['max_angle']<=.001),seconds=time.perf_counter()-t)
    rows.append(row)
    with (OUT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.save(OUT/f'parameters_fixed_{k:02d}.npy',p)
    print(f"{k+1:02d}/34 {name:15} weak={abs(v[k]):.3e} delete={same['rank']}/33 refit={post['rank']}/33 cost={result.cost:.4f} pos={forward['max_pos']:.5f} angle={forward['max_angle']:.6f} success={result.success} sec={row['seconds']:.1f}",flush=True)
(OUT/'run_metadata.json').write_text(json.dumps({'total_seconds':time.perf_counter()-start,'n_forward':len(poses)},indent=2))
print('DONE total_seconds',round(time.perf_counter()-start,1),flush=True)

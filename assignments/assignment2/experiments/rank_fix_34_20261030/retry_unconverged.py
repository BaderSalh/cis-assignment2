"""Continue only reduced fits that hit the initial 100-evaluation cap."""
from pathlib import Path
import csv,json,time,itertools,shutil
P=Path(__file__).resolve().parent
source=(P/'run_experiment.py').read_text()
exec(compile(source.split('start=time.perf_counter();fit=')[0],str(P/'run_experiment.py'),'exec'))
rows=list(csv.DictReader((OUT/'results.csv').open()))
if not (OUT/'results_initial.csv').exists():shutil.copy2(OUT/'results.csv',OUT/'results_initial.csv')
b=json.loads((OUT/'baseline.json').read_text())
levels=[(100.,300.,500.),(-200.,0.,200.),(100.,300.,500.),(-1.3,0.,1.3),(-1.3,0.,1.3),(-2.,0.,2.)]
poses=[np.array(t) for t in itertools.product(*levels)];Ftr=g['trk'].nominal_pose()
truth=[Ftr.inv()*g['rob'].actual_ee_pose(t) for t in poses]
for row in rows:
 if row['refit_success']=='True':continue
 k=int(row['index0']);active=np.arange(34)!=k
 start=time.perf_counter();previous=np.load(OUT/f'parameters_fixed_{k:02d}.npy')
 def expand(x):
  p=np.zeros(34);p[active]=x;return p
 fit=least_squares(lambda x:fast_residuals(expand(x)),previous[active],x_scale='jac',max_nfev=1000,ftol=1e-9,xtol=1e-9,gtol=1e-9)
 p=expand(fit.x);np.testing.assert_allclose(fast_residuals(p),model.residuals(p,data),atol=2e-8,rtol=1e-9)
 jr=jacobian(p);post=diagnostics(jr[:,active]);full=diagnostics(jr)
 errors=[g['metrics'].frame_error(F,model.predict_in_tracker(t,p)) for t,F in zip(poses,truth)]
 fwd=g['metrics'].summarize_errors(errors)
 row.update(refit_success=bool(fit.success),refit_nfev=int(row['refit_nfev'])+fit.nfev,refit_cost=float(fit.cost),cost_change=float(fit.cost-b['cost']),refit_rank=post['rank'],refit_full_rank=full['rank'],refit_sigma_min=post['sigma_min'],refit_condition=post['condition'],refit_normalized_rank=post['normalized_rank'],rms_pos=fwd['rms_pos'],max_pos=fwd['max_pos'],rms_angle=fwd['rms_angle'],max_angle=fwd['max_angle'],forward_pass=bool(fwd['max_pos']<=.2 and fwd['max_angle']<=.001),seconds=float(row['seconds'])+time.perf_counter()-start)
 np.save(OUT/f'parameters_fixed_{k:02d}.npy',p)
 with (OUT/'results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(row['parameter'],'success',fit.success,'nfev total',row['refit_nfev'],'cost',fit.cost,'rank',post['rank'],'pos',fwd['max_pos'],'seconds',time.perf_counter()-start,flush=True)

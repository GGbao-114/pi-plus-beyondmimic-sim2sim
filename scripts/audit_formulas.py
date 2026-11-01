import ast, hashlib, json
from pathlib import Path
import numpy as np

FILES = [
    '/data/gaoli/logs/model_align_r3/isaac_goal_dynamics_step.py',
    '/data/gaoli/logs/model_align_r3/mujoco_goal_dynamics_step.py',
    '/data/gaoli/logs/model_align_r10/mujoco_closedloop_v2.py',
    '/data/gaoli/logs/model_align_r11/isaac_v2_1a.py',
    '/data/gaoli/logs/model_align_r11/isaac_v2_1b.py',
    '/data/gaoli/logs/model_align_r8/old_closedloop_r6.py',
    '/data/gaoli/logs/model_align_r11/isaac_v2_1c.py',
    '/data/gaoli/logs/model_align_r12/phantom_stage.py',
]
S = np.load('/data/gaoli/logs/model_align_r2/isaac_goal_contact/T4_goal_hold.npz')
qs = S['base_quat'][[99,124,149,174,199]]
out=[]
for fn in FILES:
    p=Path(fn); row={'file':fn,'exists':p.exists()}
    if not p.exists(): out.append(row); continue
    text=p.read_text(errors='replace'); row['sha256']=hashlib.sha256(text.encode()).hexdigest()
    tree=ast.parse(text); funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'R','rotation','quat_matrix','quat_matrix'}]
    row['functions']=[]
    for f in funcs:
        try:
            ns={'np':np}; exec(compile(ast.Module(body=[f],type_ignores=[]),fn,'exec'),ns); fun=ns[f.name]
            errs=[]
            for q in qs:
                m=np.asarray(fun(q),dtype=float); errs.append(float(np.max(np.abs(m.T@m-np.eye(3)))))
            row['functions'].append({'name':f.name,'orthogonality_max':max(errs),'source':ast.unparse(f)})
        except Exception as e: row['functions'].append({'name':f.name,'error':repr(e)})
    row['status']='clean' if row['functions'] and all(x.get('orthogonality_max',1)>-1 and x.get('orthogonality_max',1)<1e-6 for x in row['functions']) else ('dirty' if row['functions'] else 'no_rotation_function')
    out.append(row)
Path('/data/gaoli/logs/model_align_r13').mkdir(parents=True,exist_ok=True)
Path('/data/gaoli/logs/model_align_r13/formula_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps([{'file':x['file'],'status':x.get('status'),'functions':[(f.get('name'),f.get('orthogonality_max')) for f in x.get('functions',[])]} for x in out],indent=2))

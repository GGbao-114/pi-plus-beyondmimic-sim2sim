"""Validate and summarize matched no-reset Isaac/MuJoCo traces locally."""
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'sim2sim复测_20261007'
OUT=ROOT/'isaac_hold_20261007'
MOTIONS=Path(__file__).resolve().parent/'Mini-Pi-Plus_BeyondMimic/source/motion/hightorque/pi_plus/npz'


def direction(q):
    q=np.asarray(q,dtype=np.float64)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=q.T
    return np.stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)),axis=-1)


def read_trace(path):
    with np.load(path) as source:
        return {key:source[key].copy() for key in source.files}


def summarize(data,reference):
    t=data['time']
    fi=np.minimum(np.rint(t*50).astype(int),len(reference['joint_pos'])-1)
    crossings=np.flatnonzero(data['heights']<0.15)
    fall=float(t[crossings[0]]) if len(crossings) else None
    prefall=t<fall if fall is not None else np.ones(len(t),dtype=bool)
    end=(len(reference['joint_pos'])-1)/50
    phases={'reference':prefall & (t<=end+1e-9),'hold':prefall & (t>end+1e-9)}
    jerr=np.abs(data['joint_pos']-reference['joint_pos'][fi]).mean(axis=1)
    up=direction(data['base_quat'])
    ref_up=direction(reference['body_quat_w'][fi,0])
    angular_error=np.degrees(np.arccos(np.clip((up*ref_up).sum(axis=1),-1,1)))
    result=dict(first_fall_s=fall,reference_end_s=end,min_height=float(data['heights'].min()))
    for phase,mask in phases.items():
        if not mask.any():
            result[phase]=None
            continue
        result[phase]=dict(samples=int(mask.sum()),joint_mae=float(jerr[mask].mean()),
                          up_error_max_deg=float(angular_error[mask].max()),
                          joint_speed_max=float(np.abs(data['joint_vel'][mask]).max()))
    return result


def main():
    rows=json.loads((OUT/'results/results.json').read_text())
    assert len(rows)==8 and all(r['status']=='OK' for r in rows)
    config=json.loads((OUT/'results/config.json').read_text())
    assert config['observation_noise'] is False
    report=[]
    plots=importlib.util.find_spec('matplotlib') is not None
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    for row in rows:
        if row['protocol']!='hold':
            continue
        assert row['state_writes']=={'root':0,'joint':0}
        assert not row['resample_events'] and row['obs_error_max']<1e-4
        native=read_trace(OUT/f'results/{row["run"]}_hold.npz')
        sampled=read_trace(OUT/f'results/{row["run"]}_resample.npz')
        mj_variant='updated_15s' if 'kungfu' in row['motion'] else 'updated_raw_gyro'
        mj=read_trace(ROOT/f'results/{row["run"]}_{mj_variant}.npz')
        with np.load(MOTIONS/row['motion']) as source:
            reference={key:source[key].copy() for key in ('joint_pos','joint_vel','body_pos_w','body_quat_w')}
        np.testing.assert_allclose(row['initial_state']['joint_pos'],reference['joint_pos'][0],atol=1e-5,rtol=0)
        np.testing.assert_allclose(row['initial_state']['root_state'][:3],reference['body_pos_w'][0,0],atol=1e-5,rtol=0)
        np.testing.assert_allclose(row['initial_state']['joint_vel'],0,atol=1e-5)
        np.testing.assert_allclose(row['initial_state']['root_state'][7:],0,atol=1e-5)
        expected_frame=np.minimum(np.arange(1,len(native['time'])+1),len(reference['joint_pos'])-1)
        np.testing.assert_array_equal(native['reference_frame'],expected_frame)
        # Both engines' policy-facing initial observation is fixed by the same reference and reset state.
        expected=np.zeros(119,np.float32)
        expected[:44]=np.concatenate((reference['joint_pos'][0],reference['joint_vel'][0]))
        expected[44:50]=np.eye(3)[:,:2].reshape(-1)
        expected[53:75]=reference['joint_pos'][0]-np.asarray(config['default_joint_pos'])
        np.testing.assert_allclose(native['obs'][0],expected,atol=1e-5,rtol=0)
        record=dict(run=row['run'],motion=row['motion'],isaac=summarize(native,reference),
                    mujoco=summarize(mj,reference),resample_writes=next(r['state_writes'] for r in rows if r['run']==row['run'] and r['protocol']=='resample'))
        report.append(record)
        print(json.dumps(record),flush=True)
        if plots:
            fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
            for trace,label in [(mj,'MuJoCo hold'),(native,'Isaac hold'),(sampled,'Isaac resample (state reset)')]:
                axes[0].plot(trace['time'],trace['heights'],label=label)
                q=trace['base_quat']/np.linalg.norm(trace['base_quat'],axis=1,keepdims=True)
                tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
                axes[1].plot(trace['time'],tilt,label=label)
            end=record['isaac']['reference_end_s']
            for ax in axes:
                ax.axvline(end,color='black',linestyle='--',label='reference end')
                ax.grid(alpha=0.3)
            axes[0].axhline(0.15,color='red',linestyle=':',label='fall threshold')
            axes[0].set_ylabel('Base height (m)')
            axes[1].set_ylabel('Absolute base tilt (deg)')
            axes[1].set_xlabel('Time (s)')
            axes[0].legend(loc='best',fontsize=8)
            fig.suptitle(row['motion'])
            fig.tight_layout()
            fig.savefig(OUT/f'{Path(row["motion"]).stem}_hold_comparison.png',dpi=150)
            plt.close(fig)
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['motion','reference_end_s','mujoco_first_fall_s','isaac_first_fall_s','mujoco_inmotion_mae','isaac_inmotion_mae'])
        for record in report:
            writer.writerow([record['motion'],record['isaac']['reference_end_s'],record['mujoco']['first_fall_s'],
                             record['isaac']['first_fall_s'],record['mujoco']['reference']['joint_mae'],record['isaac']['reference']['joint_mae']])
    print('COMPARISON_VALIDATED',flush=True)


if __name__=='__main__':
    main()

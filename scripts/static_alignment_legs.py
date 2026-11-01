import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path('/data/gaoli/logs/model_align_r19')
URDF=Path('/data/gaoli/projects/Mini-Pi-Plus_BeyondMimic/source/whole_body_tracking/whole_body_tracking/assets/hightorque/pi_plus/urdf/pi_plus_20dof.urdf')
MJCF=Path('/data/gaoli/logs/model_align_r13/pi_auto_fixedmass.xml')
ISAAC=Path('/data/gaoli/logs/isaac_hold_20261007/results/config.json')
TARGET=['l_thigh_link','r_thigh_link','l_calf_link','r_calf_link','l_ankle_roll_link','r_ankle_roll_link']
ISAAC_BODY_ORDER=['base_link','l_hip_pitch_link','l_shoulder_pitch_link','r_hip_pitch_link','r_shoulder_pitch_link','l_hip_roll_link','l_shoulder_roll_link','r_hip_roll_link','r_shoulder_roll_link','l_thigh_link','l_upper_arm_link','r_thigh_link','r_upper_arm_link','l_calf_link','l_elbow_link','r_calf_link','r_elbow_link','l_ankle_pitch_link','l_wrist_link','r_ankle_pitch_link','r_wrist_link','l_ankle_roll_link','r_ankle_roll_link']

def vec(s): return np.array([float(x) for x in s.replace(',',' ').split()])
def mat3(a):
    return np.array([[a[0],a[3],a[4]],[a[3],a[1],a[5]],[a[4],a[5],a[2]]],float)

ur=ET.parse(URDF).getroot(); urdf={}
for link in ur.findall('link'):
    name=link.attrib['name']; inert=link.find('inertial')
    if inert is None: continue
    org=inert.find('origin'); mass=inert.find('mass'); ine=inert.find('inertia')
    xyz=vec(org.attrib.get('xyz','0 0 0')) if org is not None else np.zeros(3)
    rpy=vec(org.attrib.get('rpy','0 0 0')) if org is not None else np.zeros(3)
    vals=[float(ine.attrib[k]) for k in ('ixx','iyy','izz','ixy','ixz','iyz')]
    I=mat3(vals); eig=np.linalg.eigvalsh(I)
    urdf[name]={'mass':float(mass.attrib['value']),'com_xyz':xyz.tolist(),'com_rpy':rpy.tolist(),'inertia':I.tolist(),'inertia_eig':eig.tolist()}

m=mujoco.MjModel.from_xml_path(str(MJCF)); mj={}
for bi in range(m.nbody):
    name=m.body(bi).name
    if name not in TARGET: continue
    I=np.asarray(m.body_inertia[bi]); eig=np.sort(I)
    mj[name]={'body_id':bi,'mass':float(m.body_mass[bi]),'com_xyz':m.body_ipos[bi].tolist(),'body_inertia_diag':I.tolist(),'body_iquat':m.body_iquat[bi].tolist(),'inertia_eig':eig.tolist(),'body_pos':m.body_pos[bi].tolist(),'body_quat':m.body_quat[bi].tolist()}

isaac=json.loads(ISAAC.read_text()); masses=np.asarray(isaac['body_masses']).reshape(-1); im={n:float(masses[i]) for i,n in enumerate(ISAAC_BODY_ORDER)}
out={}
for n in TARGET:
    u=urdf.get(n); x=mj.get(n); out[n]={'urdf':u,'mujoco':x,'isaac_mass':im.get(n),'mass_diff_mj_minus_isaac':None if x is None else x['mass']-im.get(n,0), 'mass_diff_mj_minus_urdf':None if x is None else x['mass']-u['mass']}
ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'static_alignment_legs.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

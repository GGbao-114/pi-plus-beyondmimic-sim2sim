import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--project', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--motion', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import whole_body_tracking.tasks  # noqa: F401

try:
    cfg = parse_env_cfg('Tracking-Flat-PI-Plus-Wo-v0', device=a.device, num_envs=1)
    cfg.commands.motion.motion_file = str(a.motion)
    cfg.scene.contact_forces.debug_vis = False
    cfg.scene.terrain.visual_material = None
    env = gym.make('Tracking-Flat-PI-Plus-Wo-v0', cfg=cfg)
    e = env.unwrapped
    sensor = e.scene['contact_forces']
    robot = e.scene['robot']
    result = {'attempts': [], 'objects': {}}
    def inspect(label, obj):
        names = [x for x in dir(obj) if any(k in x.lower() for k in ('contact','force','point','impulse','pair','physx'))]
        result['objects'][label] = names
        return names
    inspect('sensor', sensor); inspect('sensor_data', sensor.data); inspect('root_physx_view', robot.root_physx_view)
    for label, obj in [('sensor_data', sensor.data), ('root_physx_view', robot.root_physx_view)]:
        for name in result['objects'][label]:
            if not callable(getattr(obj, name, None)):
                try:
                    value = getattr(obj, name)
                    shape = list(value.shape) if hasattr(value, 'shape') else None
                    result['attempts'].append({'object': label, 'api': name, 'status': 'FIELD', 'type': str(type(value)), 'shape': shape, 'sample': str(value)[:500]})
                except Exception as exc:
                    result['attempts'].append({'object': label, 'api': name, 'status': 'FIELD_ERROR', 'error': repr(exc)})
                continue
            try:
                value = getattr(obj, name)()
                shape = list(value.shape) if hasattr(value, 'shape') else None
                result['attempts'].append({'object': label, 'api': name, 'status': 'OK', 'type': str(type(value)), 'shape': shape, 'sample': str(value)[:500]})
            except Exception as exc:
                result['attempts'].append({'object': label, 'api': name, 'status': 'ERROR', 'error': repr(exc)})
    try:
        import omni.physx
        result['omni_physx_import'] = 'OK'
        result['omni_physx_names'] = [x for x in dir(omni.physx) if any(k in x.lower() for k in ('contact','scene','query','interface'))]
    except Exception as exc:
        result['omni_physx_import'] = repr(exc)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'contact_api_probe.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(a.out / 'contact_api_probe.json'), 'attempts': len(result['attempts'])}, indent=2))
    env.close()
except Exception:
    traceback.print_exc(); sys.exit(1)
finally:
    app.close()

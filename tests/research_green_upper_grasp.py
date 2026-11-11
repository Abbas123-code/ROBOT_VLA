"""Read-only CPU geometry research. No simulator imports, scene or drive writes.

q is EACH finger joint position, not total jaw width. Convex hulls enclose
the source collision meshes. Surface sampling returns distance brackets:
minimum sampled signed distance minus maximum triangle edge is a lower bound.
Positive bounds additionally require that neither solid contains the other;
the tray/bottle and these small separated hulls do not have that containment.
"""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT / 'isaac/assets/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/meshes/collision'
BOTTLE = np.array([.446699321269989, -.2437739223241806, .007999688386917114])
DOWN = np.diag([1., -1., -1.])
WALLS = [
    (np.array([.354, -.25, .034]), np.array([.004, .1, .026])),
    (np.array([.546, -.25, .034]), np.array([.004, .1, .026])),
    (np.array([.45, -.346, .034]), np.array([.092, .004, .026])),
    (np.array([.45, -.154, .034]), np.array([.092, .004, .026])),
]
HAND = trimesh.load_mesh(MESH / 'hand.obj', process=True).convex_hull
FINGER_SOURCE = trimesh.load_mesh(MESH / 'finger.obj', process=True)
FINGER = FINGER_SOURCE.convex_hull


def samples(mesh, edge):
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces,
                                         max_edge=edge, max_iter=12)
    cover = float(np.linalg.norm(v[f] - np.roll(v[f], 1, axis=1), axis=2).max())
    return v, cover


def box_sdf(v, center, half):
    d = np.abs(v - center) - half
    return np.linalg.norm(np.maximum(d, 0.), axis=1) + np.minimum(d.max(axis=1), 0.)


def bottle_sdfs(v):
    p = v - BOTTLE
    radial = np.linalg.norm(p[:, :2], axis=1)
    result = []
    for z, radius, half in ((.07, .032, .07), (.17, .014, .02), (.196, .018, .006)):
        d = np.column_stack([radial - radius, np.abs(p[:, 2] - z) - half])
        result.append(np.linalg.norm(np.maximum(d, 0.), axis=1) + np.minimum(d.max(axis=1), 0.))
    result.append(np.linalg.norm(p - [0., 0., .13], axis=1) - .032)
    return np.array(result)


def vertices(name, yaw, target, h, q, offset=(0., 0.)):
    """h is ACTUAL grasp-frame height above cube ROOT, not command height."""
    r = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ DOWN
    if name == 'hand':
        v = HAND.vertices + [0., 0., -.105]
    else:
        sign = 1 if name == 'left' else -1
        v = FINGER.vertices * [sign, sign, 1.] + [0., sign * q, .0584 - .105]
    return v @ r.T + np.array([target[0] - offset[0], target[1] - offset[1], target[2] + h])


def swept_mesh(name, yaw, target, h, q0, q1=None, dz=0., offset=(0., 0.)):
    a = vertices(name, yaw, target, h, q0, offset)
    b = vertices(name, yaw, target, h, q0 if q1 is None else q1, offset) + [0., 0., dz]
    return trimesh.convex.convex_hull(np.vstack([a, b]))


def evaluate(v, cover):
    bottle = bottle_sdfs(v).min(axis=1)
    wall = np.array([box_sdf(v, c, h) for c, h in WALLS]).min(axis=0)
    return {'bottle': float(bottle.min() - cover),
            'wall': float(wall.min() - cover),
            'floor': float(v[:, 2].min() - .008),
            'bottle_upper': float(bottle.min()), 'wall_upper': float(wall.min()),
            'cover': cover}


def search(edge=.006, q=.04, h=.047, seed=7, rim_min=.005, bias=0.):
    """Heuristic bounded search, never a global infeasibility proof."""
    local = {}
    for yaw in (0, 90, 180, 270):
        # Intentional bias along finger-pad length, perpendicular to closing.
        offset = Rotation.from_euler('z', yaw, degrees=True).apply([bias, 0., 0.])[:2]
        local[yaw] = [samples(swept_mesh(n, yaw, [0, 0, .008], h, .025, q,
                                       offset=offset), edge)
                      for n in ('hand', 'left', 'right')]
    results = []
    for yaw in local:
        def objective(xy):
            target = np.array([*xy, 0.])
            gaps = []
            for v, cover in local[yaw]:
                r = evaluate(v + target, cover)
                gaps.extend([r['bottle'], r['wall']])
            cube_gap = np.linalg.norm(np.maximum(np.abs(xy - BOTTLE[:2]) - .025, 0.)) - .032
            # The robot maximin objective still requires cube/bottle >=2 mm
            # when score >=12 mm. Cube/rim is enforced by the search bounds.
            return -min(*gaps, cube_gap + .010)
        opt = differential_evolution(objective,
                                     [(.383 + rim_min, .517 - rim_min),
                                      (-.317 + rim_min, -.183 - rim_min)],
                                     popsize=8, maxiter=55, tol=.002, seed=seed, polish=True)
        results.append({'yaw': yaw, 'actual_h': h, 'release_q': q,
                        'pad_length_bias': bias, 'xy': opt.x.tolist(),
                        'lower_score': -float(opt.fun)})
    return results


def certify(x, y, h, yaw, q, edge=.001, dz=.04, offset=(0., 0.)):
    result = {}
    for phase, q0, q1, lift in [('release', .025, q, 0.), ('retreat', q, q, dz),
                                ('open_high', q, .04, 0.)]:
        target = [x, y, .008 + (dz if phase == 'open_high' else 0.)]
        result[phase] = {}
        for name in ('hand', 'left', 'right'):
            mesh = swept_mesh(name, yaw, target, h, q0, q1, lift, offset)
            v, cover = samples(mesh, edge)
            result[phase][name] = evaluate(v, cover)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=float, default=.04)
    parser.add_argument('--height', type=float, default=.047)
    args = parser.parse_args()
    print(json.dumps(search(q=args.q, h=args.height), indent=2), flush=True)
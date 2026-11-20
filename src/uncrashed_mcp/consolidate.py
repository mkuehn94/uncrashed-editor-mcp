"""Merge edge-to-edge runs of plain-box triplanar rows into fewer scaled copies.

The save format stores one transform per placed object, so engine instancing or a
merged mesh cannot be persisted. What can be persisted is fewer, bigger objects:
a triplanar material is projected in world space, so a box scaled 3 x 2 looks like
the 3 x 2 tiles it replaces (texture density unchanged, the seams disappear).

Only rows the catalogue marks triplanar AND whose LOD0 hull is a plain box are
merged (slopes, corners, holed cubes would change shape). Tiles must share row,
group, rotation, scale and the coordinate along their thinnest local axis, and lie
on one lattice with the tile pitch. Greedy maximal rectangles per lattice.
Apply runs through groups' journaled edit (delete originals + spawn replacements),
so undo_edit / edit_history work on the returned transaction ids.
Units: objects carry cm / degrees (native); the public API speaks metres.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from . import agent, groups, managed, meshes
from .geom import quat_rotate, quat_rotator, rotator_from_quat

REPORT_DIR = Path('/tmp/uncrashed-perf')
CHUNK = 4000  # edit commands per journaled transaction (native cap 5000)
LIMITATIONS = [
    'Only catalogue-triplanar rows whose hull is a plain box are merged; texture stays world-projected, seams vanish.',
    'Replacements are new actors: Blueprint/material runtime state is not carried over (catalogue rows only).',
    'Fewer actors cut per-object CPU/draw cost; a merged box is culled as one, so very large merges stay visible longer.',
    'Undo with undo_edit per transaction, newest first. No edit auto-saves.',
    'Apply sends the largest-scale merge alone first, then checks every replacement pivot/scale against the request '
    '(1 cm, 0.01%); a mismatch undoes that transaction and stops.',
]


# ---------------------------------------------------------------------------
# shape test


def hull_is_box(verts, tris, origin_cm, extent_cm, tol_cm: float = 2.0, bevel_cm: float = 3.0) -> tuple[bool, str]:
    """Every face lies on the bounds (edge bevels up to bevel_cm allowed) and each bounds face is fully covered.

    Rejects slopes, corners, holes, windows, steps and recesses; verts in metres, catalogue bounds in cm."""
    if not tris:
        return False, 'no hull'
    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    tol, bevel = tol_cm / 100, bevel_cm / 100
    for i in range(3):
        if abs(lo[i] - (origin_cm[i] - extent_cm[i]) / 100) > tol or abs(hi[i] - (origin_cm[i] + extent_cm[i]) / 100) > tol:
            return False, 'hull bounds differ from catalogue bounds'
    cover, inner, total = defaultdict(float), 0.0, 0.0
    for a, b, c in tris:
        va, vb, vc = verts[a], verts[b], verts[c]
        ab = [vb[i] - va[i] for i in range(3)]
        ac = [vc[i] - va[i] for i in range(3)]
        n = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
        length = math.sqrt(sum(x * x for x in n))
        if length < 1e-9:
            continue
        total += length / 2
        planes = [(axis, side) for axis in range(3) for side, bound in ((0, lo), (1, hi))
                  if all(abs(p[axis] - bound[axis]) <= bevel for p in (va, vb, vc))]
        if not planes:
            inner += length / 2
        for axis, side in planes:
            cover[(axis, side)] += abs(n[axis]) / 2  # area projected onto that bounds face
    if total <= 0 or inner / total > 0.005:
        return False, 'faces inside the bounds (hole, window, step, slope or recess)'
    for axis in range(3):
        u, v = [i for i in range(3) if i != axis]
        face = (hi[u] - lo[u]) * (hi[v] - lo[v])
        for side in (0, 1):
            if face > 1e-6 and not 0.97 * face <= cover.get((axis, side), 0.0) <= 2.1 * face:
                return False, 'open or partial bounds face (slope, corner or cut-out)'
    return True, 'box'


def row_shapes(catalog, rows) -> dict[str, tuple[bool, str]]:
    raw = meshes.load()
    out = {}
    for row in rows:
        try:
            asset = catalog.get(row)
        except KeyError:
            out[row] = (False, 'unknown row')
            continue
        if not asset.triplanar:
            out[row] = (False, 'not triplanar: scaling would stretch its texture')
        elif asset.kind != 'mesh' or asset.bounds_estimated or not asset.source or asset.source not in raw:
            out[row] = (False, 'no measured mesh hull')
        else:
            mesh = raw[asset.source]
            out[row] = hull_is_box(mesh.verts, mesh.tris, asset.origin_cm, asset.extent_cm)
    return out


# ---------------------------------------------------------------------------
# lattice planning


def _conj(q: dict) -> dict:
    return {'X': -q['X'], 'Y': -q['Y'], 'Z': -q['Z'], 'W': q['W']}


def _qkey(q: dict) -> tuple:
    """Rounded quaternion with q ~ -q folded: first nonzero of (W, Z, Y, X) made positive AFTER rounding,
    so yaw 180 and yaw -180 (W ~ 6e-17, Z = +-1) share a key."""
    r = {k: round(q[k], 4) + 0.0 for k in ('X', 'Y', 'Z', 'W')}
    lead = next((r[k] for k in ('W', 'Z', 'Y', 'X') if r[k] != 0), 1.0)
    sign = -1 if lead < 0 else 1
    return tuple(sign * r[k] + 0.0 for k in ('X', 'Y', 'Z', 'W'))


def _planes(items: list, fixed: int, tol: float) -> list[list]:
    """Cluster tiles by their coordinate along the thin axis (chained within tol of the cluster's first)."""
    out = []
    for it in sorted(items, key=lambda it: it[1][fixed]):
        if out and it[1][fixed] - out[-1][0][1][fixed] <= tol:
            out[-1].append(it)
        else:
            out.append([it])
    return out


def local_center(o: dict, asset) -> list[float]:
    """Bounds centre in the tile's rotated frame (cm): R^-1 * world centre."""
    q = quat_rotator(*o['xf'][3:6])
    off = quat_rotate(q, [asset.origin_cm[i] * o['xf'][6 + i] for i in range(3)])
    world = [o['xf'][i] + off[i] for i in range(3)]
    return list(quat_rotate(_conj(q), world))


def merged_xf(members: list[dict], asset, nu: int, nv: int, axes: tuple[int, int]) -> list[float]:
    first = members[0]
    q = quat_rotator(*first['xf'][3:6])
    centres = [local_center(o, asset) for o in members]
    lc = [sum(c[i] for c in centres) / len(centres) for i in range(3)]
    scale = list(first['xf'][6:])
    scale[axes[0]] *= nu
    scale[axes[1]] *= nv
    world_c = quat_rotate(q, lc)
    off = quat_rotate(q, [asset.origin_cm[i] * scale[i] for i in range(3)])
    return [world_c[i] - off[i] for i in range(3)] + list(first['xf'][3:6]) + scale


def largest_rectangle(cells: set, max_u: int, max_v: int) -> tuple[int, int, int, int] | None:
    """Largest all-occupied rectangle (u0, v0, nu, nv), clipped to max_u x max_v."""
    if not cells:
        return None
    us = [c[0] for c in cells]
    vs = [c[1] for c in cells]
    u_lo, u_hi, v_lo, v_hi = min(us), max(us), min(vs), max(vs)
    width = u_hi - u_lo + 1
    heights = [0] * width
    best = (0, None)
    for v in range(v_lo, v_hi + 1):
        for k in range(width):
            heights[k] = heights[k] + 1 if (u_lo + k, v) in cells else 0
        stack = []
        for k in range(width + 1):
            h = heights[k] if k < width else 0
            start = k
            while stack and stack[-1][1] >= h:
                start, hh = stack.pop()
                w = k - start
                nu, nv = min(w, max_u), min(hh, max_v)
                if nu * nv > best[0]:
                    best = (nu * nv, (u_lo + start, v - hh + 1, nu, nv))
            stack.append((start, h))
    return best[1]


def _components(cells: set) -> list[set]:
    seen, out = set(), []
    for c in cells:
        if c in seen:
            continue
        comp, todo = set(), [c]
        seen.add(c)
        while todo:
            u, v = todo.pop()
            comp.add((u, v))
            for n in ((u + 1, v), (u - 1, v), (u, v + 1), (u, v - 1)):
                if n in cells and n not in seen:
                    seen.add(n)
                    todo.append(n)
        out.append(comp)
    return out


def _in_region(o: dict, asset, region) -> bool:
    if region is None:
        return True
    c = local_center(o, asset)
    q = quat_rotator(*o['xf'][3:6])
    w = quat_rotate(q, c)
    return region[0] <= w[0] / 100 <= region[2] and region[1] <= w[1] / 100 <= region[3]


def plan(objects: list[dict], catalog, rows: list[str] | None = None, region=None, tolerance: float = 0.02,
         max_size: float = 64.0, top_candidates: int = 10) -> dict:
    """objects: [{actor,row,group,xf(cm,deg)}]. Pure; returns rectangles to merge and leftovers."""
    if not 0.001 <= tolerance <= 0.5:
        raise ValueError('tolerance must be 0.001..0.5 m')
    if not 1 <= max_size <= 1000:
        raise ValueError('max_size must be 1..1000 m')
    if region is not None:
        if len(region) != 4 or not all(math.isfinite(float(v)) for v in region):
            raise ValueError('region needs four finite coordinates [x0, y0, x1, y1] in metres')
        x0, y0, x1, y1 = (float(v) for v in region)
        region = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    counts = defaultdict(int)
    for o in objects:
        counts[o['row']] += 1
    wanted = set(rows) if rows else {r for r in counts if r in catalog.assets and catalog.assets[r].triplanar}
    if rows:
        for r in rows:
            if r not in catalog.assets:
                raise ValueError(f'unknown row {r!r}')
            if not catalog.assets[r].triplanar:
                raise ValueError(f'{r} ({catalog.assets[r].name}) is not triplanar: a scaled copy would stretch its texture')
    shapes = row_shapes(catalog, sorted(wanted))
    tol = tolerance * 100
    buckets = defaultdict(list)
    skipped = defaultdict(int)
    considered = 0
    for o in objects:
        if o['row'] not in wanted:
            continue
        asset = catalog.assets[o['row']]
        if not _in_region(o, asset, region):
            continue
        considered += 1
        ok, _ = shapes[o['row']]
        if not ok:
            skipped[o['row']] += 1
            continue
        if min(o['xf'][6:]) <= 0:
            skipped[o['row']] += 1
            continue
        size = [2 * asset.extent_cm[i] * o['xf'][6 + i] for i in range(3)]
        fixed = min(range(3), key=lambda i: (round(size[i], 1), 0 if i == 2 else 1, i))
        axes = tuple(i for i in range(3) if i != fixed)
        lc = local_center(o, asset)
        key = (o['row'], o['group'].casefold(), _qkey(quat_rotator(*o['xf'][3:6])),
               tuple(round(s, 4) for s in o['xf'][6:]), fixed)
        buckets[key].append((o, lc, size, axes))
    merges, duplicates = [], []
    for key, items in [(k, plane) for k, all_items in buckets.items() for plane in _planes(all_items, k[4], tol)]:
        row = key[0]
        asset = catalog.assets[row]
        _, _, size, axes = items[0]
        pu, pv = size[axes[0]], size[axes[1]]
        max_u, max_v = max(1, int(max_size * 100 / pu + 1e-6)), max(1, int(max_size * 100 / pv + 1e-6))
        lattices = []  # [ref_u, ref_v, {cell: obj}]
        for o, lc, _, _ in sorted(items, key=lambda it: (it[1][axes[0]], it[1][axes[1]], it[0]['actor'])):
            for lat in lattices:
                iu = round((lc[axes[0]] - lat[0]) / pu)
                iv = round((lc[axes[1]] - lat[1]) / pv)
                if abs(lc[axes[0]] - lat[0] - iu * pu) <= tol and abs(lc[axes[1]] - lat[1] - iv * pv) <= tol:
                    break
            else:
                lat = [lc[axes[0]], lc[axes[1]], {}]
                lattices.append(lat)
                iu = iv = 0
            if (iu, iv) in lat[2]:
                duplicates.append(o)
            else:
                lat[2][(iu, iv)] = o
        for _, _, grid in lattices:
            for comp in _components(set(grid)):
                while len(comp) > 1:
                    rect = largest_rectangle(comp, max_u, max_v)
                    if rect is None or rect[2] * rect[3] < 2:
                        break
                    u0, v0, nu, nv = rect
                    nv = min(nv, max(1, (CHUNK - 1) // nu))
                    if nu * nv < 2:
                        break
                    cells = [(u, v) for u in range(u0, u0 + nu) for v in range(v0, v0 + nv)]
                    members = [grid[c] for c in cells]
                    comp -= set(cells)
                    merges.append({'row': row, 'name': asset.name, 'group': members[0]['group'], 'tiles': len(members),
                                   'grid': [nu, nv], 'axes': list(axes), 'members': members,
                                   'xf': merged_xf(members, asset, nu, nv, axes),
                                   'size_m': [round(size[axes[0]] * nu / 100, 3), round(size[axes[1]] * nv / 100, 3)]})
    removed = sum(m['tiles'] for m in merges)
    others = sorted(((n, r) for r, n in counts.items() if r not in wanted or not shapes.get(r, (False,))[0]), reverse=True)
    candidates = []
    for n, r in others[:top_candidates]:
        asset = catalog.assets.get(r)
        if r in shapes:
            reason = shapes[r][1]
        elif asset is None:
            reason = 'unknown row (not in catalogue)'
        elif not asset.triplanar:
            reason = 'not triplanar: scaling would stretch its texture'
        else:
            reason = 'excluded by rows filter'
        candidates.append({'row': r, 'name': asset.name if asset else r, 'count': n, 'not_merged_because': reason,
                           'advice': 'thin out, use fewer larger variants, or replace by hand'})
    return {'considered': considered, 'merges': merges, 'objects_removed': removed, 'objects_added': len(merges),
            'saved': removed - len(merges), 'duplicates': duplicates,
            'skipped_rows': {r: {'count': n, 'reason': shapes[r][1]} for r, n in sorted(skipped.items())},
            'manual_candidates': candidates}


def public_plan(p: dict, limit: int = 20) -> dict:
    merges = sorted(p['merges'], key=lambda m: -m['tiles'])
    brief = []
    for m in merges[:limit]:
        brief.append({'row': m['row'], 'name': m['name'], 'group': m['group'], 'tiles': m['tiles'], 'grid': m['grid'],
                      'size_m': m['size_m'], 'pivot_m': [round(m['xf'][i] / 100, 3) for i in range(3)],
                      'rotation_pitch_yaw_roll': [round(v, 4) for v in m['xf'][3:6]],
                      'scale': [round(v, 5) for v in m['xf'][6:]]})
    by_row = defaultdict(lambda: [0, 0])
    for m in p['merges']:
        by_row[m['row']][0] += m['tiles']
        by_row[m['row']][1] += 1
    return {'considered': p['considered'], 'objects_removed': p['objects_removed'], 'objects_added': p['objects_added'],
            'saved': p['saved'], 'merges': len(p['merges']),
            'by_row': {r: {'tiles': v[0], 'replacements': v[1]} for r, v in sorted(by_row.items())},
            'top_merges': brief, 'stacked_duplicates': len(p['duplicates']),
            'duplicate_examples': [{'actor': o['actor'], 'row': o['row'], 'pivot_m': [round(o['xf'][i] / 100, 3) for i in range(3)]}
                                   for o in p['duplicates'][:5]],
            'skipped_rows': p['skipped_rows'], 'manual_candidates': p['manual_candidates']}


# ---------------------------------------------------------------------------
# sources


def objects_from_map(data: dict) -> list[dict]:
    """Saved-map objects in the snapshot shape (ids are file positions, not live handles)."""
    out = []
    for entry in data.get('S_AssetsTransform') or []:
        tags = entry.get('UniqueGroupTag') or []
        for i, t in enumerate(entry.get('Transform') or []):
            tr, q, s = t.get('Translation') or {}, t.get('Rotation') or {}, t.get('Scale3D') or {}
            rot = rotator_from_quat({k: q.get(k, 1.0 if k == 'W' else 0.0) for k in ('X', 'Y', 'Z', 'W')})
            group = tags[i] if i < len(tags) and tags[i] not in ('None', '', None) else '-'
            out.append({'actor': f'file:{len(out)}', 'row': entry['Rname'], 'group': group,
                        'xf': [float(tr.get(k, 0)) for k in 'XYZ'] + list(rot) + [float(s.get(k, 1)) for k in 'XYZ']})
    return out


def load_saved(name: str) -> tuple[dict, Path]:
    from .mapfile import sanitize_name
    from .paths import maps_dir
    path = Path(name)
    if path.suffix.lower() != '.json':
        folder = sanitize_name(name)
        path = maps_dir() / folder / f'{folder}.json'
    if not path.is_file():
        raise FileNotFoundError(f'no saved map at {path}')
    return json.loads(path.read_text()), path


def _commands(merges: list[dict]) -> list[str]:
    lines = []
    for m in merges:
        lines.append(groups.edit_line('C', after={'row': m['row'], 'group': m['group'], 'xf': m['xf']}))
        lines.extend(groups.edit_line('D', o) for o in m['members'])
    return lines


def _chunks(merges: list[dict], limit: int | None = None) -> list[list[dict]]:
    limit = CHUNK if limit is None else limit
    out, cur, n = [], [], 0
    for m in merges:
        cost = m['tiles'] + 1
        if cur and n + cost > limit:
            out.append(cur)
            cur, n = [], 0
        cur.append(m)
        n += cost
    if cur:
        out.append(cur)
    return out


POS_TOL_CM = 1.0     # replacement pivot vs requested
SCALE_REL_TOL = 1e-4  # replacement Scale3D vs requested (relative; float32 at 100x is ~1e-5)


def _close(want: dict, got: dict) -> bool:
    if want['row'].casefold() != got['row'].casefold() or want['group'].casefold() != got['group'].casefold():
        return False
    if any(abs(want['xf'][i] - got['xf'][i]) > POS_TOL_CM for i in range(3)):
        return False
    if any(abs(want['xf'][i] - got['xf'][i]) > SCALE_REL_TOL * max(1.0, abs(want['xf'][i])) for i in range(6, 9)):
        return False
    qa, qb = quat_rotator(*want['xf'][3:6]), quat_rotator(*got['xf'][3:6])
    return abs(sum(qa[k] * qb[k] for k in qa)) > 0.99999


def verify_replacements(merges: list[dict], after: list[dict]) -> list[dict]:
    """Each merge must have its own scaled copy in the post-edit state; returns the missing ones with the nearest
    object of the same row/group (pivot/scale as it came out)."""
    pool = defaultdict(list)
    for o in after:
        pool[(o['row'].casefold(), o['group'].casefold())].append(o)
    missing = []
    for m in merges:
        want = {'row': m['row'], 'group': m['group'], 'xf': m['xf']}
        bucket = pool[(m['row'].casefold(), m['group'].casefold())]
        i = next((i for i, o in enumerate(bucket) if _close(want, o)), None)
        if i is not None:
            bucket.pop(i)
            continue
        near = min(bucket, key=lambda o: sum((o['xf'][k] - m['xf'][k]) ** 2 for k in range(3)), default=None)
        missing.append({'row': m['row'], 'tiles': m['tiles'], 'requested_pivot_cm': [round(v, 1) for v in m['xf'][:3]],
                        'requested_scale': [round(v, 5) for v in m['xf'][6:]],
                        'nearest_pivot_cm': None if near is None else [round(v, 1) for v in near['xf'][:3]],
                        'nearest_scale': None if near is None else [round(v, 5) for v in near['xf'][6:]]})
    return missing


def _after(done: dict, affected: list[str]) -> list[dict]:
    try:
        return json.loads(Path(done['journal']).read_text())['after']['objects']
    except (KeyError, OSError, ValueError, TypeError):
        return groups.scoped(groups.snapshot(), affected)['objects']


def _canary_first(merges: list[dict]) -> list[list[dict]]:
    """Apply the merge with the largest scale alone first, so a Spawn Asset that clamps or ignores big
    non-uniform scales is caught (and undone) after one replacement instead of thousands."""
    if len(merges) < 2:
        return _chunks(merges)
    canary = max(merges, key=lambda m: (max(m['xf'][6:]), m['tiles']))
    return [[canary]] + _chunks([m for m in merges if m is not canary])


def consolidate_scenery(catalog, name: str = '', region=None, rows: list[str] | None = None, preview: bool = True,
                        tolerance: float = 0.02, max_size: float = 64.0, limit: int = 20) -> dict:
    if not 1 <= limit <= 200:
        raise ValueError('limit must be 1..200')
    if name:
        if not preview:
            raise ValueError('a saved map file is preview-only; load_live it, then apply on the open editor without a name')
        data, path = load_saved(name)
        p = plan(objects_from_map(data), catalog, rows, region, tolerance, max_size)
        return _report(p, limit, {'source': str(path), 'preview': True})
    with agent.session():
        snap = groups.snapshot()
        p = plan(snap['objects'], catalog, rows, region, tolerance, max_size)
        result = _report(p, limit, {'source': 'live', 'preview': preview})
        if preview or not p['merges']:
            return result
        info, root = managed.context()
        transactions = []
        for chunk in _canary_first(p['merges']):
            if transactions:
                snap = groups.snapshot()
            live = {o['actor']: o for o in snap['objects']}
            for m in chunk:
                for o in m['members']:
                    if o['actor'] not in live or not groups.same_object(o, live[o['actor']]):
                        raise RuntimeError(f'object {o["actor"]} changed during consolidation; applied={transactions}')
            affected = sorted({m['group'] for m in chunk})
            try:
                done = groups._apply(info, root, groups.scoped(snap, affected), affected, _commands(chunk),
                                     f'consolidate {len(chunk)} merges')
            except Exception as exc:
                raise RuntimeError(f'{exc}; already applied transactions={transactions} (undo newest first)') from exc
            missing = verify_replacements(chunk, _after(done, affected))
            if missing:
                try:
                    groups.undo_edit(done['transaction'], preview=False)
                    undone = f'transaction {done["transaction"]} was undone'
                except Exception as exc:  # report, never hide, a failed rollback
                    undone = f'AUTO-UNDO FAILED ({exc}); run undo_edit {done["transaction"]} with apply (preview off)'
                raise RuntimeError(f'{len(missing)} of {len(chunk)} scaled replacements did not come out as requested '
                                   f'(first: {missing[0]}); {undone}; earlier verified transactions={transactions}')
            transactions.append(done['transaction'])
        result.update(transactions=transactions, preview=False, verified=len(p['merges']),
                      undo='undo_edit <transaction> with apply (preview off) for each id, newest first')
        return result


def _report(p: dict, limit: int, extra: dict) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / 'consolidate-plan.json'
    full = {k: v for k, v in p.items() if k != 'duplicates'}
    full['duplicates'] = [{'actor': o['actor'], 'row': o['row'], 'xf': o['xf']} for o in p['duplicates']]
    path.write_text(json.dumps(full, indent=1))
    return {**extra, 'units': 'metres', **public_plan(p, limit), 'plan_json': str(path), 'limitations': LIMITATIONS}

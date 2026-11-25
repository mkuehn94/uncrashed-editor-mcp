"""How provider data enters a survey.

terrain   a terrain model (DTM, e.g. by_dgm1) replaces the scan's ground model where it has data. Its vertical
          offset to the scan (datum, geoid) is fitted on open ground (median) and reported with the residual
          spread before replacement, so the scan's own ground model is validated, not silently overwritten.
surface   a surface model (DSM, e.g. glo30) only cross-checks the scan: the scan surface averaged per provider cell
          against the provider value (offset and spread).
landcover class shares in the box against the scan classes; tree evidence per object.
buildings footprints are aligned to the scan roofs (shift search, as OSM), rasterised and paired with the mesh
          buildings by overlap. Override providers (LoD2, 3D BAG) replace footprint, heights, roof type and roof
          planes (the mesh values stay under `mesh`), split mesh buildings that hold several of their buildings, add
          buildings the mesh shows only as canopy, and flag mesh buildings they do not know. Other building
          providers (Overture) add evidence only. Trees, cars and other small objects inside a provider footprint are
          flagged as conflicts (roof greenery, carports).

Every object a provider has something to say about gets `evidence`: [{src, verdict agree | conflict | none,
detail}], and `conflicts` (count). Nothing here reaches the network: it reads the providers' caches.
"""
from __future__ import annotations

import math

import numpy as np

from .. import grid, native
from .. import osm as osm_mod
from . import geo

HEIGHT_TOL_M, HEIGHT_TOL = 1.5, 0.12
MIN_UNION_IOU = 0.25               # provider footprints (together) vs the mesh building they replace
ROOF_TOL_M, ROOF_FIT = 1.0, 0.5     # a provider roof must lie within 1 m of the scan surface on half its footprint
SMALL = ('tree', 'car', 'truck', 'shrub', 'small_object', 'low_structure', 'solar_panel')


def cell_latlon(g, georef: dict) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = g.to_world(np.arange(g.w), np.arange(g.h))
    X, Y = np.meshgrid(xs, ys)
    return geo.to_latlon(X, Y, georef['lat'], georef['lon'])


def terrain_prior(g, georef: dict, loaded: dict) -> tuple[np.ndarray | None, str | None]:
    """The first terrain provider's heights on the survey grid (its own datum; NaN where it has no data)."""
    for name, (p, data) in loaded.items():
        if p.kind == 'terrain' and data is not None:
            lat, lon = cell_latlon(g, georef)
            z = data.sample(lat, lon).astype(np.float32)
            if np.isfinite(z).mean() > 0.05:
                return z, name
    return None, None


def apply_terrain(dtm_scan: np.ndarray, dsmf: np.ndarray, valid: np.ndarray, prior: np.ndarray, name: str
                  ) -> tuple[np.ndarray, dict]:
    """(new DTM, report). Offset fitted on open ground of the scan's own model (height above it < 0.25 m)."""
    ndsm = dsmf - dtm_scan
    have = np.isfinite(prior)
    gy, gx = np.gradient(dtm_scan)
    open_ = valid & have & (ndsm < 0.25) & (np.hypot(gx, gy) < 0.15)
    if open_.sum() < 200:
        return dtm_scan, {'source': name, 'used': False, 'why': f'only {int(open_.sum())} open-ground cells overlap'}
    diff = (dtm_scan - prior)[open_]
    off = float(np.median(diff))
    res = np.abs(diff - off)
    out = np.where(have, prior + off, dtm_scan)
    out = np.minimum(out, dsmf).astype(np.float32)
    clipped = have & valid & (prior + off > dsmf + 0.3)
    rep = {'source': name, 'used': True, 'offset_m': round(off, 3), 'open_cells': int(open_.sum()),
           'residual_p50_m': round(float(np.percentile(res, 50)), 3),
           'residual_p90_m': round(float(np.percentile(res, 90)), 3),
           'residual_over_0_5m': round(float((res > 0.5).mean()), 3),
           'replaced_share': round(float((have & valid).sum() / max(1, valid.sum())), 3),
           'mesh_below_terrain_share': round(float(clipped.sum() / max(1, valid.sum())), 4),
           'change_under_objects_p90_m': round(float(np.percentile(np.abs(out - dtm_scan)[valid & ~open_ & have], 90)), 2)
           if (valid & ~open_ & have).any() else 0.0,
           'meaning': 'scan height = provider height + offset_m; residuals on open ground before replacement'}
    return out, rep


def surface_check(g, georef: dict, dsmf: np.ndarray, valid: np.ndarray, name: str, raster) -> dict:
    """Scan surface averaged per provider cell vs the provider surface model."""
    lat, lon = cell_latlon(g, georef)
    x0, y0, sx, sy = raster.transform
    if raster.epsg != 4326:
        return {'source': name, 'used': False, 'why': f'EPSG:{raster.epsg} surface check not supported'}
    c = np.floor((lon - x0) / sx).astype(int)
    r = np.floor((y0 - lat) / sy).astype(int)
    H, W = raster.array.shape
    ok = valid & (c >= 0) & (c < W) & (r >= 0) & (r < H)
    idx = (r * W + c)[ok]
    cnt = np.bincount(idx, minlength=H * W)
    mean = np.bincount(idx, dsmf[ok], minlength=H * W) / np.maximum(cnt, 1)
    full = cnt >= 0.9 * cnt.max() if cnt.max() else cnt > 0
    if full.sum() < 3:
        return {'source': name, 'used': False, 'why': 'fewer than 3 provider cells inside the survey'}
    prov = raster.array.ravel().astype(float)
    d = mean[full] - prov[full]
    off = float(np.median(d))
    return {'source': name, 'used': True, 'cells': int(full.sum()), 'offset_m': round(off, 2),
            'spread_p50_m': round(float(np.median(np.abs(d - off))), 2),
            'spread_p90_m': round(float(np.percentile(np.abs(d - off), 90)), 2),
            'meaning': 'scan surface mean per provider cell = provider + offset_m (datum check, not used for ground)'}


def landcover_check(g, georef: dict, cls: np.ndarray, valid: np.ndarray, name: str, raster) -> dict:
    lat, lon = cell_latlon(g, georef)
    wc = raster.sample(lat, lon, nearest=True)
    ok = valid & np.isfinite(wc)
    if not ok.any():
        return {'source': name, 'used': False, 'why': 'no land-cover cell inside the survey'}
    w = wc[ok].astype(int)
    c = cls[ok]
    def share(m): return round(float(m.mean()), 3)   # noqa: E704
    veg, bld = c == 11, c == 10
    built = np.isin(c, (1, 6, 7, 8, 10, 12))
    rows = {'tree_cover': [share(w == 10), share(veg)], 'built_up': [share(w == 50), share(built)],
            'grass_crop': [share(np.isin(w, (30, 40))), share(c == 2)], 'water': [share(w == 80), share(c == 3)]}
    agree_veg = round(float((w[veg] == 10).mean()), 3) if veg.any() else None
    return {'source': name, 'used': True, 'share_provider_vs_scan': rows,
            'scan_canopy_in_tree_cover': agree_veg, 'scan_buildings_in_built_up': round(float((w[bld] == 50).mean()), 3)
            if bld.any() else None,
            'meaning': 'class shares in the survey box: [provider, scan]; 10 m cells, street trees fall in built_up'}


# ---------------------------------------------------------------------------- buildings


def _scan_rings(rec: dict, georef: dict) -> list[np.ndarray]:
    rings = rec.get('rings') or ([rec['ring']] if rec.get('ring') else [])
    out = []
    for r in rings:
        a = np.asarray(r, float)
        if len(a) >= 3:
            e, s = geo.to_scan(a[:, 0], a[:, 1], georef['lat'], georef['lon'])
            out.append(np.stack([e, s], 1))
    return out


def _plane(pts: np.ndarray) -> dict | None:
    """Newell normal of a 3D ring (east, south, up) -> slope, downhill direction, area."""
    p = np.asarray(pts, float)
    if len(p) < 3:
        return None
    q = np.roll(p, -1, 0)
    n = np.array([np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
                  np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
                  np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1]))])
    ln = float(np.linalg.norm(n))
    if ln < 1e-9:
        return None
    n /= ln
    if n[2] < 0:
        n = -n
    slope = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
    horiz = math.hypot(n[0], n[1])
    faces = (math.degrees(math.atan2(n[1], n[0])) + 360) % 360 if horiz > 1e-6 else None
    return {'slope_deg': round(slope, 1), 'faces_deg': None if faces is None or slope < 2 else round(faces, 1),
            'area_m2': round(ln / 2, 1)}


def _planes(rec: dict, georef: dict, dz: float) -> list[dict]:
    out = []
    for pl in rec.get('planes') or []:
        a = np.asarray(pl, float)
        if len(a) < 3:
            continue
        e, s = geo.to_scan(a[:, 0], a[:, 1], georef['lat'], georef['lon'])
        pts = np.stack([e, s, a[:, 2] + dz], 1)
        info = _plane(pts)
        if info is None:
            continue
        info.update(z_min=round(float(pts[:, 2].min()), 2), z_max=round(float(pts[:, 2].max()), 2),
                    ring=[[round(float(x), 2), round(float(y), 2), round(float(z), 2)] for x, y, z in pts])
        out.append(info)
    return out


def _ridge(planes: list[dict], roof: str | None) -> float | None:
    sl = [p for p in planes if p['faces_deg'] is not None and p['slope_deg'] >= 5]
    if not sl or roof not in ('gabled', 'mono_pitch'):
        return None
    big = max(sl, key=lambda p: p['area_m2'])
    return round((big['faces_deg'] + 90) % 180, 1)


def _roof_from_planes(planes: list[dict], fallback):
    """Roof word and slope summary measured from the planes: flat when every plane is under 5 deg."""
    if not planes:
        return fallback, None
    area = sum(p['area_m2'] for p in planes) or 1.0
    slope = sum(p['slope_deg'] * p['area_m2'] for p in planes) / area
    if all(p['slope_deg'] < 5 for p in planes):
        return 'flat', round(slope, 1)
    return fallback, round(slope, 1)


def _ev(o: dict, src: str, verdict: str, detail: str) -> None:
    ev = o.setdefault('evidence', [])
    ev[:] = [e for e in ev if e['src'] != src]
    ev.append({'src': src, 'verdict': verdict, 'detail': detail})
    o['conflicts'] = sum(1 for e in ev if e['verdict'] == 'conflict')


class BuildingFusion:
    """One building provider against the survey (an object holding g, objects, objid, dtm, ndsm, valid, ...)."""

    def __init__(self, sv, georef: dict, name: str, provider, recs: list[dict]):
        self.sv, self.georef, self.name, self.p, self.recs = sv, georef, name, provider, recs or []

    def _rasterise(self, shift) -> tuple[np.ndarray, list[list[np.ndarray]]]:
        g = self.sv.g
        lab = np.zeros((g.h, g.w), np.int32)
        rings_all = []
        for i, rec in enumerate(self.recs, 1):
            rings = [r + shift for r in rec['_rings']]
            rings_all.append(rings)
            for r in rings:
                c, rr = g.to_cell(r[:, 0], r[:, 1])
                native.fill_poly(lab, c, rr, i)
        return lab, rings_all

    def run(self) -> dict:
        sv, g = self.sv, self.sv.g
        x1, y1 = g.x0 + g.w * g.res, g.y0 + g.h * g.res
        keep = []
        for rec in self.recs:
            rings = _scan_rings(rec, self.georef)
            if not rings:
                continue
            allp = np.vstack(rings)
            if allp[:, 0].max() < g.x0 or allp[:, 0].min() > x1 or allp[:, 1].max() < g.y0 or allp[:, 1].min() > y1:
                continue
            rec = dict(rec, _rings=rings)
            keep.append(rec)
        self.recs = keep
        rep = {'source': self.name, 'buildings_in_box': len(keep)}
        if not keep:
            return {**rep, 'used': False, 'why': 'no provider building inside the survey'}
        roofs = sv.valid & (sv.ndsm > 2.2) & (sv.exg_s < 0.02)
        feats = {'buildings': [{'osm': str(rec.get('ref')), 'rings': [r.tolist() for r in rec['_rings']]} for rec in keep]}
        shift, before, after = osm_mod.align(feats, g, roofs)
        lab, rings_all = self._rasterise(np.array(shift))
        rep.update(shift_m=[round(shift[0], 2), round(shift[1], 2)], roof_overlap=[round(before, 2), round(after, 2)])
        # vertical offset provider -> scan from building ground heights
        dz_list = []
        n = len(keep)
        flat = lab.ravel()
        order = np.argsort(flat, kind='stable')
        bounds = np.searchsorted(flat[order], np.arange(n + 2))
        dtm_flat = sv.dtm.ravel()
        for i, rec in enumerate(keep, 1):
            cells = order[bounds[i]:bounds[i + 1]]
            if rec.get('ground_h') is None or len(cells) < 8:
                continue
            dz_list.append(float(np.median(dtm_flat[cells])) - float(rec['ground_h']))
        dz = float(np.median(dz_list)) if dz_list else None
        rep['vertical_offset_m'] = None if dz is None else round(dz, 3)
        rep['vertical_offset_spread_m'] = round(float(np.median(np.abs(np.array(dz_list) - dz))), 3) if dz_list else None
        # overlap counts
        bids = [o['id'] for o in sv.objects if o['cls'] == 'building']
        objid = sv.objid
        size_l = np.bincount(lab.ravel(), minlength=n + 1)
        isb = np.zeros(len(sv.objects) + 1, bool)
        isb[np.array(bids, int)] = True
        both = (lab > 0) & (objid >= 0)
        both &= isb[np.maximum(objid, 0)]
        pairs = lab[both].astype(np.int64) * (len(sv.objects) + 1) + objid[both]
        uniq, cnt = np.unique(pairs, return_counts=True)
        inter = {}
        for u, c in zip(uniq, cnt):
            inter[(int(u // (len(sv.objects) + 1)), int(u % (len(sv.objects) + 1)))] = int(c)
        size_s = np.bincount(objid[objid >= 0], minlength=len(sv.objects))
        best_s: dict[int, tuple[int, int]] = {}
        for (li, si), c in inter.items():
            if c > best_s.get(li, (None, 0))[1]:
                best_s[li] = (si, c)
        # roof planes against the scan surface: a footprint whose roof the mesh does not show (outdated, misaligned,
        # demolished / rebuilt) must not replace the mesh geometry
        self.outside: set[int] = set()
        self.fit = self._roof_fit(keep, lab, dz, np.array(shift)) if dz is not None else {}
        fits = [v for v in self.fit.values() if v is not None]
        rep['roof_fit_p50'] = round(float(np.median(fits)), 2) if fits else None
        bad = {li for li, v in self.fit.items() if v is not None and v < ROOF_FIT}
        for li in range(1, len(keep) + 1):   # mostly beyond the scan's coverage: nothing to check it against
            if self.fit.get(li) is None and size_l[li] and sv.valid[lab == li].mean() < 0.6:
                self.fit[li] = 0.0
                self.outside.add(li)
                bad.add(li)
        rep['roof_disagrees'] = len(bad)
        why: dict[str, int] = {}
        for li in bad:   # why the scan does not show that roof
            m = lab == li
            if li in self.outside:
                k = 'outside_scan'
            elif getattr(sv, 'veg_px', np.zeros_like(m))[m].mean() >= 0.3:
                k = 'under_canopy'   # the roof cannot be judged: the provider building is trusted
                self.fit[li] = None
            elif (sv.ndsm[m] > 2.2).mean() < 0.3:
                k = 'no_roof_in_scan'
            else:
                k = 'roof_differs'
            why[k] = why.get(k, 0) + 1
        rep['roof_disagrees_why'] = why
        bad = {li for li in bad if self.fit[li] is not None}
        rep['roof_disagrees'] = len(bad)
        by_s: dict[int, list[int]] = {}
        self.bad_by_s: dict[int, list[int]] = {}
        for li, (si, c) in best_s.items():
            if c >= 0.4 * size_l[li] or c >= 0.5 * size_s[si]:
                (self.bad_by_s if li in bad else by_s).setdefault(si, []).append(li)
        matched_l = {li for ls in by_s.values() for li in ls} | bad
        rep.update(paired=len(matched_l) - len(bad), mesh_buildings=len(bids))
        if self.p.override:
            self._override(rep, keep, rings_all, lab, by_s, matched_l, inter, size_l, size_s, dz, shift)
        else:
            self._evidence(rep, keep, lab, inter, size_l, size_s, dz)
        self._small_objects(rep, keep, lab)
        rep['used'] = True
        return rep

    # -- helpers
    def _roof_fit(self, keep, lab, dz: float, shift) -> dict[int, float | None]:
        """Per provider building: share of its footprint cells where a roof plane (scan datum) is within ROOF_TOL_M
        of the scan surface. None when it has no roof planes."""
        sv, g = self.sv, self.sv.g
        self.outside = set()
        roof = np.full((g.h, g.w), -np.inf, np.float32)
        for rec in keep:
            for pl in rec.get('planes') or []:
                a = np.asarray(pl, float)
                if len(a) < 3:
                    continue
                e, s_ = geo.to_scan(a[:, 0], a[:, 1], self.georef['lat'], self.georef['lon'])
                pts = np.stack([e + shift[0], s_ + shift[1], a[:, 2] + dz], 1)
                q = np.roll(pts, -1, 0)
                n = np.array([np.sum((pts[:, 1] - q[:, 1]) * (pts[:, 2] + q[:, 2])),
                              np.sum((pts[:, 2] - q[:, 2]) * (pts[:, 0] + q[:, 0])),
                              np.sum((pts[:, 0] - q[:, 0]) * (pts[:, 1] + q[:, 1]))])
                if abs(n[2]) < 1e-6 * max(1e-9, np.linalg.norm(n)) or abs(n[2]) / np.linalg.norm(n) < 0.17:
                    continue    # (near) vertical: not seen from above
                c, r = g.to_cell(pts[:, 0], pts[:, 1])
                c0, c1 = max(0, int(np.floor(c.min()))), min(g.w - 1, int(np.ceil(c.max())))
                r0, r1 = max(0, int(np.floor(r.min()))), min(g.h - 1, int(np.ceil(r.max())))
                if c0 > c1 or r0 > r1:
                    continue
                win = np.zeros((r1 - r0 + 1, c1 - c0 + 1), np.int32)
                native.fill_poly(win, c - c0, r - r0, 1)
                rr, cc = np.nonzero(win)
                if not len(rr):
                    continue
                x, y = g.to_world(cc + c0, rr + r0)
                z = pts[0, 2] - (n[0] * (x - pts[0, 0]) + n[1] * (y - pts[0, 1])) / n[2]
                cur = roof[rr + r0, cc + c0]
                roof[rr + r0, cc + c0] = np.maximum(cur, z.astype(np.float32))
        self.roof_z = roof
        out = {}
        flat, order = lab.ravel(), None
        order = np.argsort(flat, kind='stable')
        bounds = np.searchsorted(flat[order], np.arange(len(keep) + 2))
        rz, ds, va = roof.ravel(), sv.dsmf.ravel(), sv.valid.ravel()
        for li in range(1, len(keep) + 1):
            cells = order[bounds[li]:bounds[li + 1]]
            if len(cells) and va[cells].mean() < 0.6:   # mostly beyond the scan: nothing to check against
                out[li] = 0.0
                self.outside.add(li)
                continue
            have = cells[np.isfinite(rz[cells]) & va[cells]]
            if not keep[li - 1].get('planes') or len(have) < max(8, 0.3 * len(cells)):
                out[li] = None
                continue
            out[li] = round(float((np.abs(ds[have] - rz[have]) <= ROOF_TOL_M).mean()), 3)
        return out

    def _prov_heights(self, rec: dict, ground_z: float, dz: float | None) -> dict:
        out = {'height_m': rec.get('height_m')}
        if dz is not None and rec.get('top_h') is not None:
            out['top_z'] = rec['top_h'] + dz
            out['height_m'] = out['top_z'] - ground_z
            if rec.get('eave_h') is not None:
                out['eave_m'] = rec['eave_h'] + dz - ground_z
        return out

    def _compare(self, o: dict, rec: dict, iou: float, h: dict) -> tuple[list[str], list[str]]:
        agree, conflict = [], []
        if h.get('height_m') is not None and o.get('height_m') is not None:
            d = o['height_m'] - h['height_m']
            tol = max(HEIGHT_TOL_M, HEIGHT_TOL * abs(h['height_m']))
            (agree if abs(d) <= tol else conflict).append(f"height {o['height_m']:.1f} vs {h['height_m']:.1f} m")
        (agree if iou >= 0.5 else conflict).append(f'footprint IoU {iou:.2f}')
        pr, mr = rec.get('roof'), o.get('roof')
        if pr and mr and pr != 'complex' and mr != 'complex':
            (agree if pr == mr else conflict).append(f'roof {mr} vs {pr}')
        if rec.get('storeys') and o.get('osm', {}) and str((o.get('osm') or {}).get('levels', '')).isdigit():
            lv = int(o['osm']['levels'])
            (agree if abs(lv - int(rec['storeys'])) <= 1 else conflict).append(f'storeys osm {lv} vs {rec["storeys"]}')
        return agree, conflict

    def _cells(self, lab, li):
        r, c = np.nonzero(lab == li)
        return r, c

    def _evidence(self, rep, keep, lab, inter, size_l, size_s, dz):
        sv = self.sv
        n_agree = n_conf = 0
        best_l: dict[int, tuple[int, float]] = {}
        for (li, si), c in inter.items():
            iou = c / (size_l[li] + size_s[si] - c)
            if iou > best_l.get(si, (0, 0.0))[1]:
                best_l[si] = (li, iou)
        for o in sv.objects:
            if o['cls'] != 'building':
                continue
            if o['id'] not in best_l or best_l[o['id']][1] < 0.2:
                _ev(o, self.name, 'none', 'no matching footprint')
                continue
            li, iou = best_l[o['id']]
            rec = keep[li - 1]
            h = self._prov_heights(rec, o['ground_z'], dz)
            agree, conflict = self._compare(o, rec, iou, h)
            verdict = 'conflict' if conflict else 'agree'
            n_agree += verdict == 'agree'
            n_conf += verdict == 'conflict'
            _ev(o, self.name, verdict, '; '.join(conflict or agree)[:120])
        rep.update(agree=n_agree, conflict=n_conf)

    def _override(self, rep, keep, rings_all, lab, by_s, matched_l, inter, size_l, size_s, dz, shift):
        sv, g, res = self.sv, self.sv.g, self.sv.g.res
        objs = sv.objects
        n_over = n_split = n_added = n_flag = n_agree = n_conf = n_frag = n_kept = 0
        unknown = 0
        # share of each mesh building inside provider footprints, and the provider building holding most of it
        inside: dict[int, tuple[int, int, int]] = {}
        for (li, si), c in inter.items():
            tot, best, bc = inside.get(si, (0, 0, 0))
            inside[si] = (tot + c, li if c > bc else best, max(c, bc))
        orphans = []
        for o in list(objs):
            if o['cls'] != 'building':
                continue
            ls = sorted(by_s.get(o['id'], []), key=lambda li: -size_l[li])
            if not ls:
                orphans.append(o)
                continue
            ci = sum(inter.get((lj, o['id']), 0) for lj in ls)
            union_iou = ci / max(1, size_s[o['id']] + sum(int(size_l[lj]) for lj in ls) - ci)
            if union_iou < MIN_UNION_IOU:   # the footprints hardly overlap the mesh building: do not trust either
                _ev(o, self.name, 'conflict', f'its footprints overlap the mesh building only {union_iou:.0%} (IoU): '
                                             'mesh geometry kept')
                n_kept += 1
                continue
            li = ls[0]
            c = inter.get((li, o['id']), 0)
            iou = c / (size_l[li] + size_s[o['id']] - c)
            a, cf = self._take(o, keep[li - 1], rings_all[li - 1], lab, li, dz, iou)
            n_over += 1
            n_agree += not cf
            n_conf += bool(cf)
            if len(ls) > 1:
                n_split += 1
                o['split_by'] = self.name
                for lj in ls[1:]:
                    new = self._new(keep[lj - 1], rings_all[lj - 1], lab, lj, dz, f'split from mesh building {o["id"]}')
                    if new is not None:
                        n_added += 1
                # parts of a split building go to the provider building holding their centre
                for pid in o.get('parts') or []:
                    p = objs[pid]
                    c_, r_ = g.to_cell(p['x'], p['y'])
                    owner = int(lab[min(g.h - 1, max(0, int(round(float(r_))))), min(g.w - 1, max(0, int(round(float(c_)))))])
                    if owner and owner != li:
                        tgt = next((q for q in objs if q.get('_from') == owner), None)
                        if tgt is not None:
                            p['parent'] = tgt['id']
                            tgt.setdefault('parts', []).append(pid)
                o['parts'] = [pid for pid in o.get('parts') or [] if objs[pid].get('parent') == o['id']]
        for li in range(1, len(keep) + 1):
            if li in matched_l or size_l[li] * res * res < 15:
                continue
            m = lab == li
            high = float((sv.valid[m] & (sv.ndsm[m] > 2.2)).mean())
            if high >= 0.4:
                canopy = float((sv.veg_px[m]).mean()) if hasattr(sv, 'veg_px') else 0.0
                why = 'mesh shows canopy here' if canopy >= 0.3 else 'mesh building not separated'
                if self._new(keep[li - 1], rings_all[li - 1], lab, li, dz, why) is not None:
                    n_added += 1
            else:
                unknown += 1
        for o in orphans:
            badl = self.bad_by_s.get(o['id'])
            if badl:
                refs = ', '.join(str(keep[b - 1].get('ref')) for b in badl[:3])
                fit = min(self.fit[b] for b in badl)
                what = 'lies mostly beyond the scan' if all(b in self.outside for b in badl) else \
                    f'has a roof the scan does not show (fit {fit:.0%}): outdated or rebuilt?'
                _ev(o, self.name, 'conflict', f'its building {refs} {what}; mesh geometry kept')
                n_kept += 1
                continue
            tot, li, _bc = inside.get(o['id'], (0, 0, 0))
            share = tot / max(1, size_s[o['id']])
            owner = next((q for q in objs if q.get('_from') == li), None) if li else None
            if share >= 0.5 and owner is not None:
                # a piece of a provider building the mesh survey cut off (height step): a part of its owner
                o.update(cls='building_part', parent=owner['id'], src=o['src'] + '+' + self.name, conf=owner['conf'])
                owner.setdefault('parts', []).append(o['id'])
                _ev(o, self.name, 'agree', f"part of building {owner.get('ref')} ({share:.0%} inside)")
                n_frag += 1
            elif o.get('area_m2', 0) >= 12:
                _ev(o, self.name, 'conflict', f'no building of this provider here ({share:.0%} inside one): tree, '
                                             'shelter, demolished?')
                o['conf'] = round(max(0.1, o['conf'] - 0.25), 2)
                n_flag += 1
        for o in objs:
            o.pop('_from', None)
        rep.update(overridden=n_over, split=n_split, added=n_added, fragments_to_parts=n_frag, kept_roof_disagrees=n_kept,
                   mesh_only_flagged=n_flag, agree=n_agree, conflict=n_conf, provider_only_not_in_mesh=unknown)

    def _take(self, o: dict, rec: dict, rings: list[np.ndarray], lab, li: int, dz, iou: float):
        """Override building o by provider record rec (its cells lab == li)."""
        sv, g = self.sv, self.sv.g
        r, c = self._cells(lab, li)
        if len(r) < 4:
            return [], []
        mesh = {k: o.get(k) for k in ('footprint', 'height_m', 'top_z', 'eave_m', 'ridge_m', 'roof', 'ridge_yaw_deg',
                                      'x', 'y', 'length_m', 'width_m', 'yaw_deg', 'area_m2')}
        ground = round(float(np.median(sv.dtm[r, c])), 2)
        h = self._prov_heights(rec, ground, dz)
        tmp = dict(o, ground_z=ground)
        agree, conflict = self._compare(tmp, rec, iou, h)
        fit = self.fit.get(li)
        if fit is not None:
            agree.append(f'roof planes fit the scan surface on {fit:.0%}')
            o['roof_fit'] = fit
        cx, cy, L, W, yaw = grid.obb(c, r)
        x, y = g.to_world(cx, cy)
        main = max(rings, key=lambda q: grid.ring_area(q))
        planes = _planes(rec, self.georef, dz) if dz is not None else []
        roof, slope = _roof_from_planes(planes, rec.get('roof') or o.get('roof'))
        o.update(x=round(float(x), 2), y=round(float(y), 2), ground_z=ground, length_m=round(L * g.res, 2),
                 width_m=round(W * g.res, 2), yaw_deg=round(yaw, 1), area_m2=round(len(r) * g.res ** 2, 1),
                 footprint=[[round(float(a), 2), round(float(b), 2)] for a, b in main],
                 footprint_src=self.name, geometry_src=self.name, ref=rec.get('ref'), mesh=mesh)
        if h.get('top_z') is not None:
            o.update(top_z=round(h['top_z'], 2), height_m=round(h['height_m'], 2), ridge_m=round(h['height_m'], 2))
        if h.get('eave_m') is not None:
            o['eave_m'] = round(h['eave_m'], 2)
        if roof:
            o['roof'] = roof
            o['roof_conf'] = 0.95
        if slope is not None:
            o['roof_slope_deg'] = slope
        o['ridge_yaw_deg'] = _ridge(planes, o.get('roof')) if planes else o.get('ridge_yaw_deg')
        o['roof_planes'] = planes
        if rec.get('storeys'):
            o['storeys'] = rec['storeys']
        if rec.get('function'):
            o['function'] = rec['function']
        if self.name not in o['src']:
            o['src'] = o['src'] + '+' + self.name
        if not conflict:
            o['conf'] = max(o['conf'], 0.95)
        _ev(o, self.name, 'conflict' if conflict else 'agree', '; '.join(conflict or agree)[:120])
        o['_from'] = li
        self._stamp(o, lab, li)
        return agree, conflict

    def _new(self, rec, rings, lab, li, dz, why):
        sv, g = self.sv, self.sv.g
        r, c = self._cells(lab, li)
        if len(r) < 8:
            return None
        nd = sv.ndsm[r, c]
        ground = float(np.median(sv.dtm[r, c]))
        hm = float(np.percentile(nd, 95))
        cx, cy, L, W, yaw = grid.obb(c, r)
        x, y = g.to_world(cx, cy)
        o = {'id': len(sv.objects), 'cls': 'building', 'conf': 0.8, 'src': self.name, 'x': round(float(x), 2),
             'y': round(float(y), 2), 'ground_z': round(ground, 2), 'top_z': round(ground + hm, 2),
             'height_m': round(hm, 2), 'length_m': round(L * g.res, 2), 'width_m': round(W * g.res, 2),
             'yaw_deg': round(yaw, 1), 'area_m2': round(len(r) * g.res ** 2, 1),
             'rgb': [int(v) for v in np.median(sv.rgb[r, c], 0)], 'roof': rec.get('roof'), 'roof_conf': 0.95,
             'ridge_yaw_deg': None, 'eave_m': round(float(np.percentile(nd, 15)), 2), 'ridge_m': round(hm, 2),
             'footprint': [], 'parts': [], 'osm': None, 'split_by_osm': False, 'note': why}
        mesh_h = o['height_m']
        self._take(o, rec, rings, lab, li, dz, 1.0)
        o['mesh'] = {'height_m': mesh_h}
        o['src'] = self.name
        sv.objects.append(o)
        return o

    def _stamp(self, o: dict, lab, li: int):
        r, c = self._cells(lab, li)
        cur = self.sv.objid[r, c]
        isb = np.array([q['cls'] in ('building', 'building_part') for q in self.sv.objects] + [True], bool)
        free = (cur < 0) | isb[np.minimum(np.maximum(cur, 0), len(isb) - 1)]
        self.sv.objid[r[free], c[free]] = o['id']

    def _small_objects(self, rep, keep, lab):
        sv, g = self.sv, self.sv.g
        inner = grid.erode(lab > 0, max(3, int(round(1.0 / g.res)) | 1))
        n = 0
        for o in sv.objects:
            if o['cls'] not in SMALL:
                continue
            c, r = g.to_cell(o['x'], o['y'])
            ci, ri = int(round(float(c))), int(round(float(r)))
            if not (0 <= ri < g.h and 0 <= ci < g.w):
                continue
            if inner[ri, ci]:
                li = int(lab[ri, ci])
                ref = keep[li - 1].get('ref') if li else None
                _ev(o, self.name, 'conflict', f'inside building {ref} (roof greenery / carport?)'[:120])
                cap = 0.3 if o['cls'] == 'tree' else 0.4
                o['conf'] = round(min(o['conf'], cap), 2)
                n += 1
        rep['small_objects_inside_footprints'] = n


def osm_evidence(objects: list[dict]) -> None:
    for o in objects:
        if o['cls'] == 'building' and o.get('osm'):
            _ev(o, 'osm', 'agree', f"footprint {o['osm'].get('osm')}")
        elif o['cls'] == 'tree' and o.get('src') == 'mesh+osm':
            _ev(o, 'osm', 'agree', 'tree node within 2.5 m')


def tree_landcover(objects: list[dict], g, georef: dict, raster, name: str) -> int:
    n = 0
    trees = [o for o in objects if o['cls'] == 'tree']
    if not trees:
        return 0
    xs = np.array([o['x'] for o in trees])
    ys = np.array([o['y'] for o in trees])
    lat, lon = geo.to_latlon(xs, ys, georef['lat'], georef['lon'])
    v = raster.sample(lat, lon, nearest=True)
    for o, k in zip(trees, v):
        if k == 10:
            _ev(o, name, 'agree', 'tree cover (10 m)')
            n += 1
    return n


def fuse_survey(sv, georef: dict, loaded: dict) -> dict:
    """Everything after the geometry survey (terrain already applied inside it). Returns the fusion report."""
    rep: dict = {}
    if getattr(sv, 'terrain_info', None):
        rep['terrain'] = sv.terrain_info
    osm_evidence(sv.objects)
    for name, (p, data) in loaded.items():
        if data is None:
            continue
        if p.kind == 'surface':
            rep.setdefault('surface', []).append(surface_check(sv.g, georef, sv.dsmf, sv.valid, name, data))
        elif p.kind == 'landcover':
            lc = landcover_check(sv.g, georef, sv.cls, sv.valid, name, data)
            lc['trees_in_tree_cover'] = tree_landcover(sv.objects, sv.g, georef, data, name)
            rep['landcover'] = lc
    order = sorted((kv for kv in loaded.items() if kv[1][0].kind == 'buildings' and kv[1][1]),
                   key=lambda kv: not kv[1][0].override)
    for name, (p, data) in order:
        rep.setdefault('buildings', []).append(BuildingFusion(sv, georef, name, p, data).run())
    for o in sv.objects:
        if o.get('conflicts'):
            o['flag_sources'] = ', '.join(f"{e['src']}: {e['detail']}" for e in o['evidence']
                                          if e['verdict'] == 'conflict')[:160]
    rep['objects_with_evidence'] = sum(1 for o in sv.objects if o.get('evidence'))
    rep['objects_with_conflicts'] = sum(1 for o in sv.objects if o.get('conflicts'))
    return rep

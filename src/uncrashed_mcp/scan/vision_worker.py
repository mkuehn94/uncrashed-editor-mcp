"""Vision inference for scan_survey, run by `.venv-vision/bin/python vision_worker.py JOB.json OUT.json`.

Standalone (stdlib, numpy, PIL, torch, ultralytics, transformers, cv2): it never imports uncrashed_mcp, so the
main environment needs none of these. Images are files in /dev/shm written by the caller; the worker only reads
them and writes detections in pixel coordinates. Each model is loaded once, used for every image of its phase,
then freed before the next phase (GPU memory).

Phases (each optional in the job):
  aerial.obb     YOLO11-OBB (DOTA) on ortho tiles: oriented vehicle boxes and other DOTA classes
  aerial.coarse  the same model on coarse tiles (large classes: courts, fields, pools, tanks)
  aerial.dino    Grounding DINO on ortho tiles with text prompts (car, van, ..., swimming pool)
  facades        Grounding DINO on facade renders (window, door, balcony, garage door)
  street         Grounding DINO on street-level renders (doors, windows, signs, lamps, benches, bins, ...)
  aerial.sam     SAM 2.1 masks for DINO boxes of the listed classes that no OBB box covers (oriented outline)
Models live in models_dir (downloaded on first use when missing).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

HF = {'grounding-dino-base': 'IDEA-Research/grounding-dino-base',
      'owlv2-base-patch16-ensemble': 'google/owlv2-base-patch16-ensemble'}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def pick_device(want: str):
    import torch
    if want and want not in ('auto', ''):
        return want
    if not torch.cuda.is_available():
        return 'cpu'
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        free = torch.cuda.mem_get_info(i)[0]
        if '3090' in name:  # the machine's compute card; the other drives the display and the game
            return f'cuda:{i}'
        if free > best_free:
            best, best_free = i, free
    return f'cuda:{best}'


def free(*objs) -> None:
    import torch
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def yolo_weights(models: Path, name: str) -> str:
    path = models / name
    if not path.is_file():
        from ultralytics.utils.downloads import attempt_download_asset
        models.mkdir(parents=True, exist_ok=True)
        log(f'downloading {name} to {models}')
        got = Path(attempt_download_asset(str(path)))
        if got.resolve() != path.resolve() and got.is_file():
            got.replace(path)
    return str(path)


def hf_dir(models: Path, name: str) -> str:
    path = models / name
    if not (path / 'config.json').is_file():
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        log(f'downloading {HF[name]} to {path}')
        AutoProcessor.from_pretrained(HF[name]).save_pretrained(path)
        AutoModelForZeroShotObjectDetection.from_pretrained(HF[name]).save_pretrained(path)
    return str(path)


def run_obb(paths: list[str], weights: str, device: str, imgsz: int, conf: float, batch: int = 8) -> list[list[dict]]:
    from ultralytics import YOLO
    model = YOLO(weights)
    out = []
    dev = 0 if device.startswith('cuda') else 'cpu'
    if device.startswith('cuda:'):
        dev = int(device.split(':')[1])
    for i in range(0, len(paths), batch):
        res = model.predict(paths[i:i + batch], imgsz=imgsz, conf=conf, verbose=False, device=dev)
        for r in res:
            dets = []
            if r.obb is not None:
                for c, s, p in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxyxyxy.tolist()):
                    dets.append({'class': r.names[int(c)], 'score': round(float(s), 3),
                                 'obb': [[round(float(x), 1) for x in q] for q in p]})
            out.append(dets)
    free(model)
    return out


def run_dino(groups: list[tuple[list[str], list[str], float]], weights: str, device: str, batch: int = 8,
             half: bool = True) -> list[list[list[dict]]]:
    """groups: (image paths, prompts, threshold). One model load for every group."""
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(weights)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(weights).to(device).eval()
    results = []
    for paths, prompts, thr in groups:
        text = '. '.join(p.lower() for p in prompts) + '.'
        out: list = [None] * len(paths)
        sizes = []
        for p in paths:
            with Image.open(p) as im:
                sizes.append(im.size)
        # similar aspect ratios share a batch (the processor pads a batch to its largest image)
        order = sorted(range(len(paths)), key=lambda i: sizes[i][0] / max(1, sizes[i][1]))
        i = 0
        while i < len(order):
            j = i + 1
            ar0 = sizes[order[i]][0] / max(1, sizes[order[i]][1])
            while j < len(order) and j - i < batch and sizes[order[j]][0] / max(1, sizes[order[j]][1]) < ar0 * 1.25:
                j += 1
            idx = order[i:j]
            imgs = [Image.open(paths[k]).convert('RGB') for k in idx]
            inp = proc(images=imgs, text=[text] * len(imgs), return_tensors='pt').to(device)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=half and device.startswith('cuda')):
                o = model(**inp)
            o.logits = o.logits.float()
            o.pred_boxes = o.pred_boxes.float()
            # boxes are relative to the padded batch canvas: scale by canvas / valid share of each image
            pm = inp['pixel_mask']
            Hp, Wp = pm.shape[-2:]
            targets = []
            for im, m in zip(imgs, pm):
                vh, vw = int(m.any(1).sum()), int(m.any(0).sum())
                targets.append((im.height * Hp / max(1, vh), im.width * Wp / max(1, vw)))
            res = proc.post_process_grounded_object_detection(
                o, inp.input_ids, threshold=thr, text_threshold=thr, target_sizes=targets)
            for k, r in zip(idx, res):
                labels = r.get('text_labels', r.get('labels'))
                out[k] = [{'class': str(lab), 'score': round(float(s), 3), 'box': [round(float(v), 1) for v in b]}
                          for b, s, lab in zip(r['boxes'].tolist(), r['scores'].tolist(), labels)]
            del inp, o
            i = j
        results.append(out)
    free(model, proc)
    return results


def run_sam(jobs: list[tuple[str, list[list[float]]]], weights: str, device: str) -> list[list[dict]]:
    """Per (image, boxes): the mask outline polygon and its minimum-area rectangle, pixels."""
    import cv2
    import numpy as np
    from ultralytics import SAM
    model = SAM(weights)
    dev = int(device.split(':')[1]) if device.startswith('cuda:') else ('cpu' if device == 'cpu' else 0)
    out = []
    for path, boxes in jobs:
        rows = []
        if boxes:
            r = model(path, bboxes=boxes, verbose=False, device=dev)[0]
            masks = r.masks.data.cpu().numpy() if r.masks is not None else []
            for mk in masks:
                mk = mk.astype(np.uint8)
                cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cs:
                    rows.append(None)
                    continue
                c = max(cs, key=cv2.contourArea)
                (cx, cy), (w, h), ang = cv2.minAreaRect(c)
                poly = cv2.approxPolyDP(c, 1.5, True)[:, 0, :].tolist()
                rows.append({'area_px': int(mk.sum()), 'rect': [round(cx, 1), round(cy, 1), round(w, 1), round(h, 1),
                                                                round(ang, 1)], 'polygon': poly})
        out.append(rows)
    free(model)
    return out


def _covered(box, obbs) -> bool:
    x0, y0, x1, y1 = box
    a = max(1e-6, (x1 - x0) * (y1 - y0))
    for d in obbs:
        xs = [p[0] for p in d['obb']]
        ys = [p[1] for p in d['obb']]
        ix = max(0.0, min(x1, max(xs)) - max(x0, min(xs)))
        iy = max(0.0, min(y1, max(ys)) - max(y0, min(ys)))
        if ix * iy / a > 0.4:
            return True
    return False


def main(job_path: str, out_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    models = Path(job['models_dir'])
    t_all = time.time()
    out = {'timings': {}, 'errors': [], 'models': {}}
    device = pick_device(job.get('device', 'auto'))
    out['device'] = device
    if device == 'cpu' and not job.get('allow_cpu'):
        out['errors'].append('no CUDA GPU visible to torch (set UNCRASHED_VISION_CPU=1 to run on the CPU, slowly)')
        Path(out_path).write_text(json.dumps(out))
        return 0
    import torch
    out['gpu'] = torch.cuda.get_device_name(int(device.split(':')[1])) if device.startswith('cuda:') else device
    aer = job.get('aerial') or {}
    fac = job.get('facades') or {}
    stv = job.get('street') or {}
    peak = 0.0

    def mark(name, t0):
        nonlocal peak
        out['timings'][name + '_s'] = round(time.time() - t0, 2)
        if torch.cuda.is_available() and device.startswith('cuda'):
            peak = max(peak, torch.cuda.max_memory_allocated(device) / 1e9)

    if aer.get('tiles') and aer.get('obb'):
        t0 = time.time()
        o = aer['obb']
        w = yolo_weights(models, o['model'])
        out['models']['obb'] = o['model']
        out['obb'] = run_obb(aer['tiles'], w, device, o.get('imgsz', 1024), o.get('conf', 0.15))
        if aer.get('coarse'):
            out['coarse'] = run_obb(aer['coarse'], w, device, o.get('imgsz', 1024), o.get('conf', 0.15))
        mark('obb', t0)
    groups = []
    if aer.get('tiles') and aer.get('dino'):
        groups.append((aer['tiles'], aer['dino']['prompts'], aer['dino'].get('thr', 0.25)))
    if fac.get('images'):
        groups.append((fac['images'], fac['prompts'], fac.get('thr', 0.25)))
    stv_groups = [gr for gr in stv.get('groups', []) if gr.get('images')]
    for gr in stv_groups:
        groups.append((gr['images'], gr['prompts'], gr.get('thr', 0.25)))
    if groups:
        t0 = time.time()
        name = job.get('dino_model', 'grounding-dino-base')
        res = run_dino(groups, hf_dir(models, name), device)
        out['models']['dino'] = name
        if aer.get('tiles') and aer.get('dino'):
            out['dino'] = res.pop(0)
        if fac.get('images'):
            out['facades'] = res.pop(0)
        if stv_groups:
            out['street'] = [res.pop(0) for _ in stv_groups]
        mark('dino', t0)
    sam = aer.get('sam')
    if sam and out.get('dino'):
        t0 = time.time()
        want = set(sam.get('classes', []))
        jobs, index = [], []
        for ti, dets in enumerate(out['dino']):
            obbs = (out.get('obb') or [[]] * len(out['dino']))[ti]
            pick = [k for k, d in enumerate(dets) if d['class'] in want and d['score'] >= sam.get('thr', 0.3)
                    and not _covered(d['box'], obbs)]
            if pick:
                jobs.append((aer['tiles'][ti], [dets[k]['box'] for k in pick]))
                index.append((ti, pick))
        masks = run_sam(jobs, yolo_weights(models, sam.get('model', 'sam2.1_b.pt')), device) if jobs else []
        out['models']['sam'] = sam.get('model', 'sam2.1_b.pt')
        for (ti, pick), rows in zip(index, masks):
            for k, m in zip(pick, rows):
                if m:
                    out['dino'][ti][k]['mask'] = m
        out['timings']['sam_boxes'] = sum(len(p) for _, p in index)
        mark('sam', t0)
    out['timings']['total_s'] = round(time.time() - t_all, 2)
    out['vram_peak_gb'] = round(peak, 2)
    Path(out_path).write_text(json.dumps(out))
    return 0


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    os.environ.setdefault('YOLO_VERBOSE', 'False')
    sys.exit(main(sys.argv[1], sys.argv[2]))

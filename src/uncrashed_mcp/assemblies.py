"""Intent-aware support connectivity, explicitly NOT a structural/load solver."""
from __future__ import annotations

import json
import math
import tempfile
import uuid
from collections import deque
from pathlib import Path

from . import agent, environment, groups, managed
from .collide import Rows

ROLES = {'foundation','structure','cantilever','decoration'}


def define_assembly(name: str = '', roles: dict[str,str] | None = None, preview: bool = True) -> dict:
    if roles is None:
        with agent.session():
            info,root=managed.context()
            saved=managed.read_state(info,root)['assemblies']
            if name:
                managed.tag(name)
                if name not in saved:
                    raise ValueError('unknown assembly')
                return {'name':name, **saved[name]}
            return {'assemblies':saved}
    managed.tag(name)
    if not roles or len(roles)>256 or any(role not in ROLES for role in roles.values()):
        raise ValueError('roles maps 1..256 group names to foundation|structure|cantilever|decoration')
    for group in roles:
        managed.tag(group)
    if len({g.casefold() for g in roles})!=len(roles):
        raise ValueError('group names must be unique ignoring case')
    if 'foundation' not in roles.values():
        raise ValueError('declare at least one foundation group; no inferred z=0 anchor')
    with agent.session():
        info,root = managed.context()
        objects = groups.snapshot()['objects']
        existing = {o['group'].casefold() for o in objects}
        missing = [g for g in roles if g.casefold() not in existing]
        if missing:
            raise ValueError('groups missing: '+', '.join(missing))
        count = sum(o['group'].casefold() in {g.casefold() for g in roles} for o in objects)
        if count>groups.MAX_EDIT:
            raise ValueError('assembly limited to 5000 objects')
        state = managed.read_state(info,root)
        previous = state['assemblies'].get(name)
        result = {'name': name, 'roles': roles, 'objects': count, 'preview': preview, 'previous': previous}
        if not preview:
            state['assemblies'][name] = {'roles': roles}
            managed.write_state(root,state)
            result['sidecar'] = str(root/managed.STATE_NAME)
        return result


def parse_contacts(text: str, expected: set[str]) -> tuple[dict,set[str],list[str]]:
    facts, truncated, errors, count = {}, set(), [], None
    for line in text.splitlines():
        p = line.split()
        if p[0]=='A' and len(p)>=3:
            address=environment.actor_id(p[1])
            if address in facts:
                raise ValueError('duplicate contact result')
            facts[address]={environment.actor_id(v) for v in p[3:]}
            if int(p[2]):
                truncated.add(address)
        elif p[0]=='error':
            errors.append(environment.actor_id(environment.fields(line)['actor']))
        elif p[0]=='end':
            count=int(environment.fields(line)['count'])
        else:
            raise ValueError('invalid contact result')
    if count!=len(expected) or set(facts)|set(errors)!=expected or len(facts)+len(errors)!=len(expected):
        raise ValueError('incomplete assembly contacts')
    return facts,truncated,errors


def parse_anchors(text: str, expected: set[str]) -> tuple[list[dict],bool]:
    records,errors,count,truncated=[],[],None,False
    for line in text.splitlines():
        p=line.split()
        if p[0]=='F':
            f=environment.fields(line)
            records.append({'actor':environment.actor_id(p[1]), **{k:int(f[k]) for k in ('samples','anchored','missing','buried')}, 'basis':f['basis']})
        elif p[0]=='error':
            errors.append(environment.actor_id(environment.fields(line)['actor']))
        elif p[0]=='end':
            f=environment.fields(line);count=int(f['count']);truncated=bool(int(f['truncated']))
        else:
            raise ValueError('invalid anchor result')
    found=[r['actor'] for r in records]+errors
    if count!=len(expected) or set(found)!=expected or len(found)!=len(expected):
        raise ValueError('incomplete foundation anchors')
    records.extend({'actor':a,'samples':0,'anchored':0,'missing':0,'buried':0,'basis':'unavailable'} for a in errors)
    return records,truncated


def analyse(objects: list[dict], roles: dict[str,str], contacts: dict, anchors: list[dict],
            min_coverage: float, plants: set[str] | None = None) -> dict:
    plants=plants or set()
    indexed={o['actor']:o for o in objects}
    role={o['actor']:roles[o['group'].casefold()] for o in objects}
    graph={a:set() for a in indexed}
    for a,neighbors in contacts.items():
        for b in neighbors:
            if b in indexed and a in contacts.get(b,set()):
                graph[a].add(b)
    roots,foundation_warnings,unknown=set(),[],[]
    for f in anchors:
        coverage=f['anchored']/f['samples'] if f['samples'] else None
        f['coverage']=coverage
        confirmed=(f['basis']=='collision_vertical_span' and f['samples']>=3 and not f['missing'])
        if not confirmed:
            unknown.append(f['actor'])
        if confirmed and coverage>=min_coverage and f['actor'] not in plants:
            roots.add(f['actor'])
        if not confirmed or coverage is None or coverage<.95 or f['buried']:
            foundation_warnings.append(f)
    # Decorations/plants cannot turn disconnected steel into a supported frame.
    reachable=set(roots);queue=deque(sorted(roots))
    while queue:
        a=queue.popleft()
        for b in graph[a]:
            if b not in reachable and b not in plants and role[b]!='decoration':
                reachable.add(b);queue.append(b)
    disconnected=[a for a in indexed if a not in reachable and a not in plants and role[a]!='decoration']
    decorative_detached=[a for a in indexed if (role[a]=='decoration' or a in plants) and not graph[a].intersection(reachable)]
    return {'roots':sorted(roots),'reachable':sorted(reachable),'disconnected':disconnected,
            'decorative_detached':decorative_detached,'foundation_warnings':foundation_warnings,'unknown_foundations':unknown,
            'anchored_cantilevers':[a for a in reachable if role[a]=='cantilever'],
            'edges':[[a,b] for a in graph for b in sorted(graph[a]) if a<b]}


def check_assembly(catalog, name: str, tolerance: float = .05, foundation_tolerance: float = .3,
                   min_anchor_coverage: float = .5, top: int = 20) -> dict:
    managed.tag(name)
    if (not all(math.isfinite(v) for v in (tolerance,foundation_tolerance,min_anchor_coverage)) or
            not 0<=tolerance<=1 or not 0<=foundation_tolerance<=1 or not 0<min_anchor_coverage<=1 or not 1<=top<=100):
        raise ValueError('tolerances 0..1 m, coverage (0,1], top 1..100 required')
    with agent.session(),tempfile.TemporaryDirectory(prefix='uncrashed-assembly-',dir='/tmp') as tmp:
        info,root=managed.context();state=managed.read_state(info,root)
        if name not in state['assemblies']:
            raise ValueError('unknown assembly; define_assembly first')
        roles={k.casefold():v for k,v in state['assemblies'][name]['roles'].items()}
        if not roles or any(v not in ROLES for v in roles.values()) or 'foundation' not in roles.values():
            raise ValueError('invalid assembly role metadata')
        snap=groups.snapshot()
        objects=[o for o in snap['objects'] if o['group'].casefold() in roles]
        missing=set(roles)-{o['group'].casefold() for o in objects}
        if missing or not objects or len(objects)>groups.MAX_EDIT:
            raise ValueError(f'assembly missing/empty/oversized groups: {sorted(missing)}')
        foundations=[o for o in objects if roles[o['group'].casefold()]=='foundation']
        request,result=Path(tmp)/'request.txt',Path(tmp)/'result.txt'
        request.write_text('\n'.join('A '+o['actor'] for o in objects)+'\n')
        agent.run(f'contacts {agent._windows_path(request)} {agent._windows_path(result)} {tolerance*100:.6f}',120)
        contacts,truncated,errors=parse_contacts(result.read_text(),{o['actor'] for o in objects})
        request.write_text('\n'.join('A '+o['actor'] for o in foundations)+'\n')
        agent.run(f'anchors {agent._windows_path(request)} {agent._windows_path(result)} {foundation_tolerance*100:.6f}',120)
        anchors,anchor_truncated=parse_anchors(result.read_text(),{o['actor'] for o in foundations})
        rows=Rows.from_catalog(catalog)
        plants={o['actor'] for o in objects if rows.cat(o['row'])=='PLANT'}
        facts=analyse(objects,roles,contacts,anchors,min_anchor_coverage,plants)
        incomplete=bool(truncated or errors or anchor_truncated or facts['unknown_foundations'])
        review=bool(facts['disconnected'] or facts['foundation_warnings'])
        report={'name':name,'status':'incomplete' if incomplete else 'review' if review else 'no_connectivity_findings',
                'units':'metres','roles':roles,'objects':objects,'contacts':{k:sorted(v) for k,v in contacts.items()},
                'truncated_contacts':sorted(truncated),'contact_errors':errors,'anchor_truncated':anchor_truncated,
                'anchors':anchors,**facts,
                'parameters':{'contact_tolerance_m':tolerance,'foundation_tolerance_m':foundation_tolerance,'min_anchor_coverage':min_anchor_coverage},
                'limitations':['Connectivity heuristic, NOT load-bearing/stability certification or exact mesh-mesh intersection.',
                               'Edges require reciprocal oriented-bounds-versus-collision hits; bounds can bridge empty mesh space.',
                               'Only explicit foundations seed support. Bare landscape/base traces ignore placed actors and foliage.',
                               'Foundation anchors use 5x5 vertical collision spans; hollow/tilted/composite foundations need manual review.',
                               'Intentional cantilevers need an anchor path, not full ground footprint; strength/deflection are not evaluated.',
                               'Plants/decorations never propagate structural support; external assemblies are not inferred as anchors.']}
        managed.REPORTS.mkdir(parents=True,exist_ok=True)
        output=managed.REPORTS/f'assembly-{name}-{uuid.uuid4().hex}.json'
        managed.atomic_write(output,json.dumps(report,allow_nan=False))
        index={o['actor']:o for o in objects}
        return {'name':name,'status':report['status'],'objects_checked':len(objects),'anchors':len(facts['roots']),
                'connected':len(facts['reachable']),'disconnected_count':len(facts['disconnected']),
                'disconnected':[index[a] for a in facts['disconnected'][:top]],
                'anchored_cantilevers':len(facts['anchored_cantilevers']),
                'foundation_warnings':facts['foundation_warnings'][:top],
                'unknown_foundations':facts['unknown_foundations'][:top],
                'decorative_detached_count':len(facts['decorative_detached']),
                'truncated':bool(truncated or anchor_truncated),'contact_errors':errors[:top],
                'report_json':str(output),'limitations':report['limitations']}

"""Opt-in live integration test. Never clears/saves existing map.

Temporary tagged cubes + foliage masks, temporary metadata folder. Restores map
save directory and test geometry in finally. Requires running editor/new DLL.
Run: .venv/bin/python scripts/live_managed_smoke.py --allow-temporary-edits
"""
import argparse
import hashlib
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from uncrashed_mcp import agent, assemblies, groups, live, managed
from uncrashed_mcp.catalog import load_catalog


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-temporary-edits',action='store_true',required=True)
    parser.parse_args()
    report={}
    original=agent.map_info()
    original_path=managed.directory_path(original)/f'{original["folder"]}.json'
    original_hash=hashlib.sha256(original_path.read_bytes()).hexdigest()
    prefix='mcp_smoke_'+uuid.uuid4().hex[:10]
    with tempfile.TemporaryDirectory(prefix='uncrashed-smoke-',dir='/tmp') as tmp:
        root=Path(tmp)
        # Metadata sandbox only, not a copy/save of existing geometry.
        (root/f'{root.name}.json').write_text('{}')
        try:
            agent.run(f'setmap {agent._windows_path(root)}')
            commands=[agent._spawn_command('NewRow_13',1000+i*3,1000,400,13,27,7,(1,1,1),prefix+'.main') for i in range(2)]
            agent.batch(commands)
            before=groups.snapshot()
            own=lambda:[o for o in groups.snapshot()['objects'] if o['group'].startswith(prefix+'.')]
            assert len(own())==2
            edit=groups.edit_group(prefix+'.main',delta=[3,-2,1],rotation=[11,27,5],scale=1.2,preview=False)
            report['transform']=edit
            groups.undo_edit(edit['transaction'],False)
            groups.match_states([o for o in before['objects'] if o['group'].startswith(prefix+'.')],own())
            clone=groups.edit_group(prefix+'.main',action='clone',delta=[20,0,0],target_group=prefix+'.copy',preview=False)
            assert len(own())==4
            groups.undo_edit(clone['transaction'],False)
            assert len(own())==2
            assign=groups.edit_group(prefix+'.main',action='assign',actors=[own()[0]['actor']],target_group=prefix+'.part',preview=False)
            assert any(o['group']==prefix+'.part' for o in own())
            groups.undo_edit(assign['transaction'],False)
            delete=groups.edit_group(prefix+'.main',action='delete',preview=False)
            assert not own()
            groups.undo_edit(delete['transaction'],False)
            assert len(own())==2
            report['clone_assign_delete_undo']='passed'
            # Simulate reload of only our objects: new handles, same saved state.
            edit=groups.edit_group(prefix+'.main',delta=[1,0,0],preview=False)
            shifted=own()
            agent.batch([f'destroy {o["actor"]}' for o in shifted])
            for o in shifted:
                x,y,z,p,yaw,r,sx,sy,sz=o['xf']
                agent.run(agent._spawn_command(o['row'],x/100,y/100,z/100,p,yaw,r,(sx,sy,sz),o['group']))
            groups.undo_edit(edit['transaction'],False)
            report['undo_after_handle_changes']='passed'
            # Terrain-grounded footing + intentional cantilever, outside district.
            terrain=agent.run('terrain 50000 0 100000 -200000')
            ground=float(managed.environment.fields(terrain[0])['z'])/100
            agent.batch([
                agent._spawn_command('NewRow_13',500,0,ground+.9,0,0,0,(4,4,2),prefix+'.foot'),
                agent._spawn_command('NewRow_13',504,0,ground+2.05,0,0,0,(10,1,.3),prefix+'.beam'),
            ])
            assemblies.define_assembly(prefix, {prefix+'.foot':'foundation',prefix+'.beam':'cantilever'},False)
            audit=assemblies.check_assembly(load_catalog(),prefix)
            report['assembly']=audit
            assert audit['anchors']==1 and audit['anchored_cantilevers']==1, audit
            branch=groups.edit_group(prefix,action='clone',prefix=True,target_group=prefix+'.dup',delta=[20,0,0],preview=False)
            assert {o['group'] for o in own() if o['group'].startswith(prefix+'.dup.')}=={prefix+'.dup.main',prefix+'.dup.foot',prefix+'.dup.beam'}
            groups.undo_edit(branch['transaction'],False)
            report['hierarchical_clone_undo']='passed'
            # A known, supported tree source. Narrow mid-crown box avoids ground.
            inventory=managed.foliage_inventory([-365,95,-45],[-340,120,0],2)
            trees=[t for t in inventory['foliage'] if not t['masked']]
            assert trees,'known LongRange1 tree absent'
            tree=trees[0];lo,hi=tree['aabb']['min'],tree['aabb']['max']
            center=[(a+b)/2 for a,b in zip(lo,hi)]
            ray=f'trace {center[0]*100:.6f} {center[1]*100:.6f} {(hi[2]+5)*100:.6f} {(lo[2]-5)*100:.6f}'
            collision_before=agent.run(ray)
            assert any('InstancedFoliageActor' in line for line in collision_before)
            preview=managed.clearing_zones('upsert','test',center,[1,1,1])
            assert preview['candidate_instances']>=1
            applied=managed.clearing_zones('upsert','test',center,[1,1,1],preview=False,allow_partial=True)
            report['clearing']=applied
            collision_masked=agent.run(ray)
            assert not any('InstancedFoliageActor' in line for line in collision_masked)
            assert any(t['masked'] for t in managed.foliage_inventory(lo,hi,2)['foliage'])
            # Union semantics: removing one overlapping zone must not restore tree.
            managed.clearing_zones('upsert','overlap',center,[1,1,1],preview=False,allow_partial=True)
            managed.clearing_zones('remove','test',preview=False)
            assert any(t['masked'] for t in managed.foliage_inventory(lo,hi,2)['foliage'])
            # Directory switch restores; returning reapplies without Python zone call.
            agent.run(f'setmap {original["directory"]}');agent.run('zones refresh')
            assert not any(t['masked'] for t in managed.foliage_inventory(lo,hi,2)['foliage'])
            agent.run(f'setmap {agent._windows_path(root)}');time.sleep(1.3)
            assert any(t['masked'] for t in managed.foliage_inventory(lo,hi,2)['foliage'])
            # Hot unload must restore originals before releasing its storage.
            assert agent.unload()['unloaded']
            time.sleep(.2)
            agent.run('zones refresh')  # injects again, reads portable config
            assert any(t['masked'] for t in managed.foliage_inventory(lo,hi,2)['foliage'])
            managed.clearing_zones('remove','overlap',preview=False)
            assert not any(t['masked'] for t in managed.foliage_inventory(lo,hi,2)['foliage'])
            collision_restored=agent.run(ray)
            assert collision_restored==collision_before
            report['zone_collision']={'before':collision_before,'masked':collision_masked,'restored':collision_restored}
            report['zone_union_reload_restore']='passed'
            focus=live._xdo('getactivewindow')
            pose=live.read_camera()
            shot=live.capture_view(pose['x'],pose['y'],pose['z'],pose['yaw'],pose['pitch'])
            report['capture']={'path':str(shot),'focus_before':focus,'focus_after':live._xdo('getactivewindow'),
                               'camera_before':pose,'camera_after':live.read_camera()}
        finally:
            managed.REPORTS.mkdir(parents=True,exist_ok=True)
            archive=managed.REPORTS/(prefix+'-state')
            shutil.copytree(root,archive)
            report['sandbox_archive']=str(archive)
            # Delete only unique test tags. Never a region-wide clear/destruction.
            for o in groups.snapshot()['objects']:
                if o['group'].startswith(prefix+'.'):
                    agent.run(f'destroy {o["actor"]}')
            (root/managed.ZONES_NAME).unlink(missing_ok=True)
            try:
                agent.run('zones refresh')
            finally:
                agent.run(f'setmap {original["directory"]}')
            agent.run('zones refresh')
            final=agent.map_info()
            report['restored']={'folder':final['folder'],'objects':final['objects'],'gates':final['gates'],
                                'original_save_unchanged':hashlib.sha256(original_path.read_bytes()).hexdigest()==original_hash}
            output=managed.REPORTS/'live-smoke.json';managed.atomic_write(output,json.dumps(report,indent=2))
            print(json.dumps(report,indent=2),flush=True)
            assert final['objects']==original['objects'] and final['gates']==original['gates']
            assert report['restored']['original_save_unchanged']


if __name__=='__main__':
    main()

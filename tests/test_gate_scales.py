"""Saved race-gate sizes survive the native unit-scale spawn used by load_live."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uncrashed_mcp import agent
from uncrashed_mcp.geom import quat_rotator


def transform(x=0, scale=1):
    return dict(Translation=dict(X=x,Y=0,Z=300),Rotation=quat_rotator(0,90,0),Scale3D=dict(X=scale,Y=scale,Z=scale))


def line(actor='1234',x=0,yaw=90,scale=1):
    return f'gate {actor} row=- class=BP_Gate_C loc={x},0,300 rot=0,{yaw},0 scale={scale},{scale},{scale}'


class GateScaleTest(unittest.TestCase):
    def test_scaled_and_unit_gates(self):
        gates=[('STARTGATE',transform()),('InflRond02',transform(100,1.8))]
        self.assertEqual(agent._gate_scale_commands(gates,[line(),line('5678',100),'count 2']),['transform 5678 - - - - - - 1.8 1.8 1.8'])

    def test_default_saved_scale(self):
        t=transform();del t['Scale3D']
        self.assertEqual(agent._gate_scale_commands([('STARTGATE',t)],[line()]),[])

    def test_changed_count_refused(self):
        with self.assertRaisesRegex(agent.AgentError,'gate count changed'):
            agent._gate_scale_commands([('STARTGATE',transform())],[])

    def test_changed_position_refused(self):
        with self.assertRaisesRegex(agent.AgentError,'gate sequence changed'):
            agent._gate_scale_commands([('STARTGATE',transform())],[line(x=1)])

    def test_changed_rotation_refused(self):
        with self.assertRaisesRegex(agent.AgentError,'gate transform changed'):
            agent._gate_scale_commands([('STARTGATE',transform())],[line(yaw=91)])

    def test_changed_scale_refused(self):
        with self.assertRaisesRegex(agent.AgentError,'gate transform changed'):
            agent._gate_scale_commands([('STARTGATE',transform())],[line(scale=2)])

    def test_nonfinite_live_transform_refused(self):
        for current in (line(x='nan'),line(scale='nan')):
            with self.subTest(current=current),self.assertRaises(agent.AgentError):
                agent._gate_scale_commands([('STARTGATE',transform())],[current])

    def test_invalid_saved_scale_refused(self):
        for value in (0,-1,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaisesRegex(agent.AgentError,'invalid saved scale'):
                agent._gate_scale_commands([('STARTGATE',transform(scale=value))],[line()])

    def test_load_applies_sizes_after_spawning(self):
        doc=SimpleNamespace(base='LongRange1',order=[],groups={},race=dict(StartTransform=transform(),CheckPoints=[dict(RowName='InflRond02',Transform=transform(100,1.8))],Laps=1),cam={},display_name='Scaled race',instance_count=lambda:0)
        info=dict(base='LongRange1',objects=0,gates=2)
        def run(command,*args):
            return [line(),line('5678',100)] if command=='gates' else []
        with patch.object(agent,'load_map',return_value=doc),patch.object(agent,'map_info',return_value=info),patch.object(agent,'clear',return_value={}),patch.object(agent,'run',side_effect=run),patch.object(agent,'batch',side_effect=[({},'spawn ok'),({},'scale ok')]) as batch:
            result=agent._load('Scaled',move_camera=False)
        self.assertEqual(result['failed_count'],0)
        self.assertEqual(batch.call_args_list[1].args[0],['transform 5678 - - - - - - 1.8 1.8 1.8'])

    def test_load_scale_failure_is_reported(self):
        doc=SimpleNamespace(base='LongRange1',order=[],groups={},race=dict(StartTransform=transform(),CheckPoints=[dict(RowName='InflRond02',Transform=transform(100,1.8))]),cam={},display_name='Scaled',instance_count=lambda:0)
        with patch.object(agent,'load_map',return_value=doc),patch.object(agent,'map_info',return_value=dict(base='LongRange1',objects=0,gates=0)),patch.object(agent,'clear',return_value={}),patch.object(agent,'run',return_value=[]),patch.object(agent,'batch',return_value=({},'spawn ok')):
            result=agent._load('Scaled',move_camera=False)
        self.assertEqual(result['failed_count'],1)
        self.assertIn('gate count changed',result['failed'][0]['error'])

if __name__=='__main__':unittest.main()

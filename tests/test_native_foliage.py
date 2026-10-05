"""Execute exact native guard used for zero-scale restoration (no UE required)."""
import ctypes
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('cc'), 'host C compiler unavailable')
class NativeFoliageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        root=Path(cls.tmp.name)
        source=root/'guard.c'
        source.write_text('#include "foliage_state.h"\nint hidden(const unsigned char*a,const unsigned char*b){return foliage_hidden_transform_matches(a,b);}\nint original(const unsigned char*a,const unsigned char*b){return foliage_original_transform_matches(a,b);}\n')
        subprocess.run(['cc','-shared','-fPIC','-O2','-I',str(Path(__file__).resolve().parents[1]/'native'),str(source),'-o',str(root/'guard.so'),'-lm'],check=True,capture_output=True)
        cls.lib=ctypes.CDLL(str(root/'guard.so'))
        cls.lib.hidden.argtypes=cls.lib.original.argtypes=[ctypes.c_void_p,ctypes.c_void_p]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def transforms(self):
        f=ctypes.c_float*12
        # Actual live failure: original rotated tree; zero matrix reads identity.
        original=f(0,0,-.6357427,.771901,-35387.921875,10597.44238,-3983.38037,0,.9999146,.9999146,.9999145,0)
        hidden=f(0,0,0,1,-35387.921875,10597.44238,-3983.38037,0,0,0,0,0)
        return original,hidden

    def test_zero_scale_loses_quaternion_but_still_restorable(self):
        original,hidden=self.transforms()
        self.assertEqual(self.lib.hidden(original,hidden),1)

    def test_different_instance_location_refuses_restoration(self):
        original,hidden=self.transforms();hidden[4]+=1
        self.assertEqual(self.lib.hidden(original,hidden),0)

    def test_nonzero_scale_refuses_mask_identity(self):
        original,hidden=self.transforms();hidden[8]=.01
        self.assertEqual(self.lib.hidden(original,hidden),0)

    def test_nan_translation_refuses_restoration(self):
        original,hidden=self.transforms();hidden[4]=float('nan')
        self.assertEqual(self.lib.hidden(original,hidden),0)

    def test_unchanged_original_setter_failure_needs_no_restore(self):
        original,_=self.transforms()
        self.assertEqual(self.lib.original(original,original),1)

    def test_arbitrary_external_edit_not_original(self):
        original,_=self.transforms();changed=(ctypes.c_float*12)(*original);changed[9]=2
        self.assertEqual(self.lib.original(original,changed),0)

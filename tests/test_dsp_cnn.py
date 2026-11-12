"""Pre-experiment unit checks; no real test labels enter these checks."""
import unittest
import numpy as np
import torch
from dsp_cnn import envelope_map, CompactCNN

class CNNTests(unittest.TestCase):
    def test_roi_dc_and_envelope_pooling(self):
        image=np.zeros((480,7168));image[:,1100:3100]=5*np.sin(2*np.pi*np.arange(2000)/100)
        expected=envelope_map(image)
        self.assertEqual(expected.shape,(60,125))
        np.testing.assert_allclose(expected,np.log1p(5),rtol=1e-6)
        image[:,:1100]=1e9;image[:,3100:]=-1e9
        image[:,1100:3100]+=np.arange(480)[:,None]
        np.testing.assert_allclose(envelope_map(image),expected,atol=1e-6)
    def test_shape_and_zero_input(self):
        with self.assertRaises(ValueError):envelope_map(np.zeros((5,5)))
        self.assertTrue(np.isfinite(envelope_map(np.zeros((480,7168)))).all())
    def test_forward_backward(self):
        model=CompactCNN();x=torch.randn(3,1,60,125)
        logits=model(x)
        self.assertEqual(tuple(logits.shape),(3,))
        torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.,0.])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':unittest.main()

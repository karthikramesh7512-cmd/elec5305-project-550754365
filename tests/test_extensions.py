import unittest
import numpy as np
import torch
from complete_extensions import tfm_reconstruct
from noise_robustness import add_noise,gpu_envelope
from scipy.signal import hilbert

class ExtensionTests(unittest.TestCase):
    def test_exact_noise_snr_and_dc(self):
        g=torch.Generator().manual_seed(17)
        rf=torch.randn((2,8,64),generator=g,dtype=torch.float64)
        rf-=rf.mean(-1,keepdim=True)
        noise=torch.randn(rf.shape,generator=g,dtype=torch.float64)
        for snr in [0,5,10,15,20]:
            delta=add_noise(rf,snr,noise)-rf
            realized=10*torch.log10(rf.square().mean((1,2))/delta.square().mean((1,2)))
            torch.testing.assert_close(realized,torch.full_like(realized,float(snr)),atol=1e-12,rtol=0)
            self.assertLess(float(delta.mean(-1).abs().max()),1e-14)
        np.testing.assert_allclose(gpu_envelope(rf).numpy(),np.abs(hilbert(rf.numpy(),axis=-1)),atol=1e-12)

    def test_point_reflector_localization(self):
        fs=20e6;c=5820.;n=8;pitch=.0007;time=np.arange(512)/fs
        truth=np.array([.0025,.012]);data=np.zeros((512,n,n))
        for i in range(n):
            for j in range(n):
                tau=(np.hypot(truth[0]-i*pitch,truth[1])+np.hypot(truth[0]-j*pitch,truth[1]))/c
                dt=time-tau
                data[:,i,j]=np.cos(2*np.pi*5e6*dt)*np.exp(-(dt/1.5e-7)**2)
        xs=np.arange(-.001,.00701,.00025);zs=np.arange(.009,.01501,.00025)
        image=tfm_reconstruct(data,xs,zs,fs,c,pitch)
        k=np.unravel_index(image.argmax(),image.shape)
        estimate=np.array([xs[k[1]],zs[k[0]]])
        self.assertLessEqual(np.linalg.norm(estimate-truth),.00026)

if __name__=='__main__':unittest.main()

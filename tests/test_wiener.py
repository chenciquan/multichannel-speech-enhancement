import unittest
import numpy as np
import torch
import torchaudio.functional as AF
from mcse.wiener import masked_covariances,wiener_weights,mask_mwf


class WienerTests(unittest.TestCase):
    def test_diagonal_scalar_solution(self):
        ps=torch.diag(torch.arange(1,7,dtype=torch.float64)).to(torch.complex128)[None];pn=2*torch.eye(6,dtype=torch.complex128)[None]
        w,_=wiener_weights(ps,pn,mu=3,relative_loading=0,absolute_loading=0);expected=torch.zeros_like(w);expected[0,4]=5/11
        torch.testing.assert_close(w,expected,rtol=1e-12,atol=1e-12)

    def test_complex_rankone_reference_phase(self):
        h=torch.tensor([1+.3j,.4-.7j,2+.2j,-.2+.4j,1-.5j,.2+.9j],dtype=torch.complex128);p=2.;sigma=.7;mu=1.3
        ps=(p*h[:,None]*h.conj()[None,:])[None];pn=sigma*torch.eye(6,dtype=torch.complex128)[None]
        w,_=wiener_weights(ps,pn,mu,0,0);expected=p*h*h[4].conj()/(mu*sigma+p*float(h.abs().square().sum()))
        torch.testing.assert_close(w[0],expected,rtol=1e-11,atol=1e-12)
        X=h[:,None,None]*torch.tensor([[[.3+.2j,.8-.4j]]],dtype=torch.complex128)
        out=AF.apply_beamforming(w,X);gain=p*float(h.abs().square().sum())/(mu*sigma+p*float(h.abs().square().sum()))
        torch.testing.assert_close(out[0],gain*X[4,0],rtol=1e-11,atol=1e-12)

    def test_constant_mask_preserves_fraction_not_half(self):
        rng=np.random.default_rng(712);X=torch.from_numpy(rng.normal(size=(6,3,40))+1j*rng.normal(size=(6,3,40)));mask=torch.full((3,40),.2,dtype=torch.float64)
        ps,pn=masked_covariances(X,mask);w,_=wiener_weights(ps,pn,1,0,0);expected=torch.zeros_like(w);expected[:,4]=.2
        torch.testing.assert_close(w,expected,rtol=1e-11,atol=1e-12)
        rx=torch.einsum('cft,dft->fcd',X,X.conj())/40;torch.testing.assert_close(ps+pn,rx,rtol=1e-11,atol=1e-12)

    def test_zero_and_rank_deficient_remain_finite(self):
        p=torch.zeros((2,6,6),dtype=torch.complex128);w,_=wiener_weights(p,p);self.assertTrue(torch.equal(w,torch.zeros_like(w)))
        p[:]=1;w,info=wiener_weights(p,p);self.assertTrue(torch.isfinite(w).all());self.assertLess(info['solve_relative_residual_max'],1e-8)
        y,_=mask_mwf(np.zeros((512,6),dtype='float32'),np.zeros(512,dtype='float32'));np.testing.assert_array_equal(y,0)

    def test_common_covariance_scale_invariance_without_absolute_floor(self):
        ps=torch.eye(6,dtype=torch.complex128)[None];pn=2*ps
        a,_=wiener_weights(ps,pn,1,1e-7,0);b,_=wiener_weights(ps*9,pn*9,1,1e-7,0);torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_invalid_input_does_not_fallback(self):
        p=torch.eye(6,dtype=torch.complex128)[None]
        with self.assertRaises(ValueError):wiener_weights(p,p,0)
        with self.assertRaises(ValueError):wiener_weights(p,-p)
        with self.assertRaises(ValueError):mask_mwf(np.ones((600,5)),np.ones(600))
        with self.assertRaises(ValueError):mask_mwf(np.ones((600,6)),np.ones(599))

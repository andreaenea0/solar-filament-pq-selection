"""Verify the minority-support criterion against the actual host denominator."""

import numpy as np
from filseg.metric import encode,score_image
from tools.figures.minority_support import contributions


def test_minority_support_can_have_positive_pq_surplus():
    prediction=np.zeros((16,16),dtype=np.uint8);prediction[2:6,2:6]=1
    truth=prediction.copy();truth[6,2:6]=1
    mask=encode(prediction)
    drawings=[[],[encode(truth)],[]]
    n,d,s,m=contributions(drawings,[mask])
    assert s.tolist()==[1] and m.tolist()==[0]
    np.testing.assert_allclose(n,[.8]);np.testing.assert_allclose(d,[1.5])
    assert n[0]-.455*d[0]>0  # One of three drawings can justify inclusion.
    actual=score_image(drawings,[mask])
    np.testing.assert_allclose(actual.pq,n[0]/(.5+d[0]))


def test_multiple_hits_use_the_generalized_denominator():
    mask=encode(np.ones((8,8),dtype=np.uint8))
    drawings=[[mask,mask]]
    n,d,s,m=contributions(drawings,[mask])
    assert (n.tolist(),d.tolist(),s.tolist(),m.tolist())==([2.],[1.],[1],[1])
    actual=score_image(drawings,[mask])
    np.testing.assert_allclose(actual.pq,n[0]/(1+d[0]))

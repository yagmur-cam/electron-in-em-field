import numpy as np 
import pytest 
from fields import uniform_field, exb_field, mirror_field, make_mirror_field, make_bottle_field, make_penning_field
from analysis import divergence

ALL_FIELDS = [uniform_field, exb_field, mirror_field, make_mirror_field(k=0.05),
    make_bottle_field(k=0.3, a=4.0),
    make_penning_field(B0=1.0, C=0.05)] #the only field with a non-zero E 


@pytest.mark.parametrize("fld", ALL_FIELDS) #only for test_divergence_B to test it for every field, without a loop 
def test_divergence_B(fld):
    assert abs(divergence(fld, [0.3, 0.4, 0.5])) < 1e-9


def test_penning():
    field = make_penning_field(B0=1.0, C=0.05)
    assert abs(divergence(field, [0.3, 0.4, 0.5], which=0)) < 1e-9


def test_uniform_field():
    E, B = uniform_field(np.array([1.0, 2.0, 3.0]), 0.0)
    np.testing.assert_array_equal(B, [0, 0, 1])  #T/F 
    np.testing.assert_array_equal(E, [0, 0, 0])


#|B| at z=a exceeds |B| at z=0 and at z=2a. That's what makes it a bottle.
def test_bottle_max():
    a = 4.0
    fld = make_bottle_field(k=0.3, a=a)
    mags = [np.linalg.norm(fld([0,0,z], 0.0)[1]) for z in (0.0,a,2*a)]
    assert mags[1] > mags[0] and mags[1] > mags[2] #mags[1] is the peak, z=a


@pytest.mark.parametrize("fld", ALL_FIELDS)
def test_field_shapes(fld):
    E, B = fld(np.array([0.3,0.4,0.5]), 0.0)
    assert E.shape == (3,)
    assert B.shape == (3,)

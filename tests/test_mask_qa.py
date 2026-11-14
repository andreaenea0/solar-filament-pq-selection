"""Check diagnostic topology, image grouping and compact-core specificity."""
import numpy as np
import pytest
from types import SimpleNamespace
from tools.figures.mask_qa import compact_cores,measure_mask,radial_coordinates


def test_topology_holes_and_overlap_use_pixels():
    mask=np.zeros((40,40),dtype=np.uint8)
    mask[5:25,5:25]=1;mask[10:15,10:15]=0;mask[30:32,30:32]=1
    occupancy=mask.astype(np.uint16);occupancy[5:7,5:7]=2
    cores=np.zeros_like(mask);cores[20:22,20:22]=1
    rho=np.ones(mask.shape)*.99;rho[30:32,30:32]=1.01
    row=measure_mask(mask,rho,occupancy,cores)
    assert row['area']==379 and row['parts']==2
    assert row['holes']==1 and row['hole_area']==row['largest_hole']==25
    assert row['overlap_pixels']==row['core_pixels']==4
    assert row['limb_pixels']==379 and row['off_disk_pixels']==4
    assert row['rho_max']==1.01 and row['rho_min']==.99


def test_diagonal_foreground_is_connected_and_empty_rejected():
    mask=np.eye(4,dtype=np.uint8)
    row=measure_mask(mask,np.zeros_like(mask),mask,np.zeros_like(mask))
    assert row['parts']==1 and row['holes']==0
    with pytest.raises(ValueError,match='empty'):
        measure_mask(mask*0,mask,mask,mask)


def test_core_proxy_keeps_dark_compact_disk_and_rejects_long_line():
    yy,xx=np.ogrid[:512,:512]
    image=np.full((512,512),180,dtype=np.uint8)
    image[(xx-220)**2+(yy-240)**2<=8**2]=35
    image[280:284,200:300]=35
    disk=SimpleNamespace(x=255.5,y=255.5,radius=230.)
    rho=radial_coordinates(image.shape,disk)
    result=compact_cores(image,rho)
    assert result[240,220]==1
    assert result[281,230]==0

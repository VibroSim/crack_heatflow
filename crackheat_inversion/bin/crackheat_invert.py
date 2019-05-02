#! /usr/bin/env python

import sys
import os
import ast

import dg_file as dgf
import pyopencl as cl
import numpy as np
from matplotlib import pyplot as pl

from ..heatinversion import heatinvert_wfm

def main(args=None):
    if args is None:
        args=sys.argv
        pass

    if len(args) < 3:
        print("Usage: crackheat_invert <dgs_file> <material> \"<cracktipcoords1>\" \"<cracktipcoords2>\" <tikparam> <rstep>")
        print(" ")
        print("The crack tip coordinates can be pasted from the oscilloscope display;")
        print("They should be symmetric about the crack center, and along the length of the crack. If they are beyond the crack tips that is OK so long as they are symmetric and aligned reasonably closely to the tips. They should be no more than a mm or two offset laterally from the crack.")
        print(" ")
        print(" material can be a tuple (k,rho,c) in MKS units ")
        print(" or Ti64 or In718 to use book value for those materials ")
        print(" ")
        print(" tikparam defaults to 2e-7 m*deg K/Joule")
        print(" rstep defaults to 1mm ")
        
        sys.exit(1)
        pass
    
    dgs_file = args[1]
    material = args[2]
    cracktip1 = args[3]
    cracktip2 = args[4]
    if len(args) >= 6:
        tikparam = float(args[5])
        pass
    else:
        tikparam = 2e-7
        pass

    if len(args) >= 7 :
        rstep = float(args[6])
        pass
    else:
        rstep=1e-3
        pass


    (metadata,wfmdict)=dgf.loadsnapshot(dgs_file)

    if "VibroFit" in wfmdict:
        HeatingChannel=wfmdict["VibroFit"]
        pass
    else:
        HeatingChannel=wfmdict["DiffStack"]
        pass
         
    if "Arb" in wfmdict:
        Arb=wfmdict["Arb"]
        pass
    else:
        Arb=wfmdict["arb"]
        pass

    if material=="Ti64":
        matl_k=6.7 # W/m/deg K
        matl_rho=4430.0 # kg/m^3
        matl_c = 526.3 #
        pass
    elif material=="In718":
        matl_k=11.4  # W/m/deg K
        matl_rho=8190.0 # kg/m^3
        matl_c=435.0 #  
    else:
        (matl_k,matl_rho,matl_c)=ast.literal_eval(material)
        matl_k=float(matl_k)
        matl_rho=float(matl_rho)
        matl_c=float(matl_c)        
        pass

    (cracktip1x,cracktip1y) = ast.literal_eval(cracktip1)
    cracktip1x=float(cracktip1x)
    cracktip1y=float(cracktip1y)
    
    (cracktip2x,cracktip2y) = ast.literal_eval(cracktip2)
    cracktip2x=float(cracktip2x)
    cracktip2y=float(cracktip2y)

    ctx=cl.create_some_context()
    
    (crack_pos,
     crack_centerpos,
     int_pos,
     int_axis,
     timebase,
     use_data,
     use_data_integrated,
     highres_idx,
     excend_idx,
     r_bnds,
     bestfit,recon,s,
     bestfit_highres,recon_highres,s_highres) = heatinvert_wfm(wfmdict,HeatingChannel,Arb,matl_k,matl_rho,matl_c,cracktip1x,cracktip1y,cracktip2x,cracktip2y,tikparam,rstep,ctx=ctx)

    r_centers=(r_bnds[:-1]+r_bnds[1:])/2.0


    d_int=int_pos[1]-int_pos[0]
    d_crack=crack_pos[1]-crack_pos[0]
    if int_axis==0:
        extent=(int_pos[0]-d_int/2.0,int_pos[-1]+d_int/2.0,crack_pos[0]-d_crack/2.0,crack_pos[-1]+d_crack/2.0)
        pass
    else:
        extent=(crack_pos[0]-d_crack/2.0,crack_pos[-1]+d_crack/2.0,int_pos[0]-d_int/2.0,int_pos[-1]+d_int/2.0)
        pass
    pl.figure()
    pl.imshow(use_data[:,:,highres_idx].T,extent=extent,origin='lower')
    pl.colorbar()
    pl.title('Crack heating (highres frame), deg. K')

    pl.figure()
    pl.imshow(use_data[:,:,excend_idx].T,extent=extent,origin='lower')
    pl.colorbar()
    pl.title('Crack heating (excitation end frame), deg. K')

    pl.figure()
    pl.subplot(2,1,1)
    pl.imshow(use_data_integrated,vmin=0,vmax=np.max(use_data_integrated)*1.1)
    pl.colorbar()
    pl.title('Crack heating, integrated (lowres)')
    
    pl.subplot(2,1,2)
    pl.imshow(recon,vmin=0,vmax=np.max(use_data_integrated)*1.1)
    pl.colorbar()
    pl.title('Inversion reconstruction (lowres)')

    pl.figure()
    pl.plot(r_centers,bestfit[::2],'-',
            r_centers,bestfit[1::2],'-')
    pl.title('recovered source intensity as function of r (lowres)')
    pl.ylabel('Source intensity, W/m^2')
    pl.legend(('x < 0','x > 0'))


    pl.figure()
    pl.plot(s)
    pl.plot((0,s.shape[0]-1),(tikparam,tikparam))
    pl.title('Tikhonov parameter diagnostic (lowres)')
    pl.legend(('Singular values','Tikhonov parameter'))
    


    
    pl.figure()
    pl.plot(crack_pos,use_data_integrated[:,highres_idx])
    pl.plot(crack_pos,recon_highres[:,0])
    pl.title('Integrated crack heating (highres frame) and reconstruction (highres)')
    pl.legend(('Measured data','Reconstruction'))
    pl.xlabel('Position along crack (m)')

    
    pl.figure()
    pl.plot(r_centers,bestfit_highres[::2],'-',
            r_centers,bestfit_highres[1::2],'-')
    pl.title('recovered source intensity as function of r (highres)')
    pl.ylabel('Source intensity, W/m^2')
    pl.xlabel('Radius (m)')
    pl.legend(('x < 0','x > 0'))
    
    pl.figure()
    pl.plot(s_highres)
    pl.plot((0,s_highres.shape[0]-1),(0.0,0.0))
    pl.title('Tikhonov parameter diagnostic (highres)')
    pl.legend(('Singular values','Tikhonov parameter'))
    
    
    sys.modules["__main__"].__dict__.update(globals())
    sys.modules["__main__"].__dict__.update(locals())
    pl.show()
    pass

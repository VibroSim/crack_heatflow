import numpy as np
from .heatpredict_accel import surface_heating_y_integral
import numpy.linalg
import scipy.optimize
import dg_metadata as dgm
import re
import dg_eval

import pyopencl as cl

# Heatinvert.py: Given heating as a function of position x along
# crack, evaluate heating

def heatinvert(x,xcenter,t,t1,t2,k,rhoc,strip_r_bnds,refpos_y,infpos_y,Tdata,tikparam,ctx=None):
    """ Invert observed heating on the assumption that the 
heating originates from quarter-circular bands centered at
xcenter. Tdata is should be a function of x (axis 0, direction along crack) 
and t (axis 1) and integrated along y (direction across crack)

Tikparam may need to be tuned if the time base changes, but should be 
invariant under spatial changes. 
"""

    if ctx is None:
        ctx=cl.create_some_context()
        pass
    
    alpha=k/float(rhoc)

    nx=x.shape[0]
    nt=t.shape[0]

    dx=x[1]-x[0]
    #if t.shape[0] > 1:
    #    dt=t[1]-t[0]
    #    pass
    
    assert((nx,nt)==Tdata.shape)
    n_bnds = strip_r_bnds.shape[0]-1
    dr = strip_r_bnds[1]-strip_r_bnds[0]

    inversion_mtx = np.zeros((nx*nt,2*n_bnds),dtype='d')
    # Each column is nx*nt long representing the integrals along y at that x,t
    # Each column represents a particular quarter circle
    
    # Create inversion matrix
    for r_idx in range(n_bnds):
        r_inner=strip_r_bnds[r_idx]
        r_outer=strip_r_bnds[r_idx+1]

        # do x < xcenter side of semicircle first
        inversion_mtx[:,r_idx*2] = surface_heating_y_integral(refpos_y,infpos_y,x[:,np.newaxis]-xcenter,t[np.newaxis,:],r_inner,r_outer,t1,t2,alpha,k,False,ctx).reshape(nx*nt)
        # x > xcenter side of semicircle
        inversion_mtx[:,r_idx*2+1] = surface_heating_y_integral(refpos_y,infpos_y,x[:,np.newaxis]-xcenter,t[np.newaxis,:],r_inner,r_outer,t1,t2,alpha,k,True,ctx).reshape(nx*nt)
        pass

    # Green's function represents heating temperature due to a unit magnitude
    # point heat impulse.
    # if convolved with a delta function in time and 3 space dimensions
    # of 1 Joule * (1/s) * (1/m^3) (integrated over s*m^3) it yields deg K. 
    # Therefore GF has units of deg K/Joule. The (1/s * 1/m^3) are the units
    # of the delta distribution of the heat impulse

    # inversion_mtx is the Green's function convolved with a unit W/m^2
    # source over excitation time and two spatial dimensions then
    # integrated over y measurement position.
    # The source has units of W/m^2 * (1/m) (integrated over s*m^3) it
    # yields deg. K. therefore inversion_mtx prior to y integration
    # has units of deg K/Joule
    # the heating area, and integrated over y measurement position
    # it has units of m*degK/Joule

    # Make Tikhonov parameter scale invariant
    # by pre-scaling A by dx... then we will divide U by dx
    # likewise we will pre-scale A by 1/dr and multiply V by dr
    # ... see greensinversion paper and greensinversion's sourcevecs.py
    # for complete rationale

    inversion_mtx_scaled = inversion_mtx * dx/dr

    
    # Do tikhonov regularized inversion with SVD
    #
    (u,s,v)=np.linalg.svd(inversion_mtx_scaled,full_matrices=False)   # inversion/t_amount

    # inversion_mtx_scaled has same units as inversion_mtx
    # because multiplied by m/m
    # ... or really m^2 * degK/(Joule-m)
    
    # Scale u and v according to row scaling and column scaling
    # inversion_mtx was  multiplied by row_scaling/column_scaling, i.e. dx/dr
    
    # so that  A_scaled (column_scaling*x) = b*row_scaling 
    # or A_scaled = A*row_scaling/column_scaling
    
    # dividing u by row scaling
    # and multiplying columns of v by column scaling
    # Would make u*s*v the equivalent of the unscaled matrix. 
    
    # But that is not what we will use u and v for...
    # Instead ut s, and vt form the inverse: 
    # vt * sinv * ut:   where x = vt * sinv * ut * b
    # We want this to apply to an unscaled vector b
    # and give an unscaled result x
    # So we need to scale the columns of ut (rows of u) 
    # by multiplying by row_scaling
    # and scale the rows of vt (columns of v)
    # by dividing by column_scalng
    
    # note that u_scaled and v_scaled are no longer orthogonal matrices
    
    u_scaled = u * dx   #/t_amount
    v_scaled = v / dr


    d=s/(s**2.0+tikparam**2.0)
    #
    bestfit = np.dot(v_scaled.T,np.dot(u_scaled.T,Tdata.reshape(nx*nt))*d)
    #
    # do NNLS regularized inversion
    # ... concatenate regularization matrix to inversion_mtx_scaled
    #inversion_mtx_scaled_cat = np.concatenate((inversion_mtx_scaled,np.sqrt(tikparam)*np.eye(inversion_mtx_scaled.shape[1])))
    #Tdata_cat = np.concatenate((Tdata.reshape(nx*nt),np.zeros(inversion_mtx_scaled.shape[1])))
    #bestfit = scipy.optimize.nnls(inversion_mtx_scaled_cat,Tdata_cat)*dx/dr # perform nnls and correct for scaling


    # Check amount of negative heating... Should be less than 30% of total
    if np.abs(np.sum(bestfit[bestfit < 0.0])) > 0.3*np.sum(bestfit[bestfit > 0.0]):
        sys.stderr.write("crackheat_inversion/heatinvert: Not enough heating for accurate evaluation (or bad Tikhonov parameter); heating values: %s; returning NaN\n" % (str(bestfit)))
        
        bestfit[bestfit < 0.0] = 0.0
        recon=np.dot(inversion_mtx,bestfit).reshape(nx,nt) # Illustrate recon
        
        bestfit[:]=np.nan # return NaN if too much negative heating
        pass
    else:
        # Set negative heatings to zero
        bestfit[bestfit < 0.0] = 0.0
        recon=np.dot(inversion_mtx,bestfit).reshape(nx,nt)
        pass
    
    return (bestfit,recon,s)  # always returns singular values from regular least squares
    

def heatinvert_wfm(wfmdict,HeatingChannel,Arb,matl_k,matl_rho,matl_c,crackcenterx,crackcentery,cracktip1x,cracktip1y,tikparam,rstep,ctx=None):

    integration_width=1e-2 # 1 cm
    extra_length=1e-3 # 1cm


    excwfm=dgm.GetMetaDatumWIStr(Arb,"WfmGenCmd","")

    # Match for %f: ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)
    
    matchobj = re.match(r"""GEN:BURST [Aa]rb ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) Hz ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) s ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) s ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) s ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) s""",excwfm)
    excfreq = float(matchobj.group(1))
    exc_t1=float(matchobj.group(2))
    exc_t2=float(matchobj.group(3))
    exc_t3=float(matchobj.group(4))
    exc_t4=float(matchobj.group(5))


    # Extract diffstack
    (ndim,DimLen,IniVal,Step,bases) = dg_eval.geom(HeatingChannel)
    (ndim,Coord,Units,AmplCoord,AmplUnits)=dg_eval.axes(HeatingChannel)

    evaluated = dg_eval.evalv(HeatingChannel,wfmdict,ndim,IniVal,Step,DimLen)

    # For the moment, we assume assume the crack is near horizontal
    # or near vertical


    crackangle = np.arctan2(cracktip1y-crackcentery,cracktip1x-crackcenterx)
    if np.abs(crackangle) < np.pi/10 or np.abs(crackangle-np.pi) < np.pi/10 or np.abs(crackangle+np.pi) < np.pi/10:
        # approximately horizontal
        # ... do vertical integration

        int_centerpos = crackcentery
        int_bottom=int_centerpos-integration_width/2.0
        int_top=int_centerpos+integration_width/2.0
        int_bottom_idx = (int_bottom-IniVal[1])/Step[1]
        int_top_idx = (int_top-IniVal[1])/Step[1]

        int_start_idx=min(int(round(int_bottom_idx)),int(round(int_top_idx)))
        int_start_idx=max(0,int_start_idx)
        int_end_idx=max(int(round(int_bottom_idx)),int(round(int_top_idx)))
        int_end_idx=min(int(DimLen[1])-1,int_end_idx)
        

        
        if cracktip1x < crackcenterx:
            crack_region = (cracktip1x-extra_length/2.0,crackcenterx + half_length +extra_length/2.0)
            pass
        else:
            half_length = cracktip1x-crackcenterx
            crack_region = (crackcenterx-half_length-extra_length/2.0,cracktip1x+extra_length/2.0)
            pass
        
        crack_region_idxs=np.array(((crack_region[0]-IniVal[0])/Step[0],(crack_region[1]-IniVal[0])/Step[0]),dtype='d')
        crack_region_start=min(int(round(crack_region_idxs[0])),int(round(crack_region_idxs[1])))
        crack_region_start=max(0,crack_region_start)
        crack_region_end=max(int(round(crack_region_idxs[0])),int(round(crack_region_idxs[1])))
        crack_region_end=min(int(DimLen[0])-1,crack_region_end)

        use_data = evaluated.data[crack_region_start:(crack_region_end+1),int_start_idx:(int_end_idx+1),:]
        
        int_axis=1

        use_data_integrated = np.sum(use_data,axis=int_axis)*Step[1] # integral over dy

        crack_pos = bases[0][crack_region_start:(crack_region_end+1)]
        crack_length=abs(cracktip1x-crackcenterx)*2.0
        int_pos=bases[1][int_start_idx:(int_end_idx+1)]

        crack_centerpos = crackcenterx
        
        pass
    elif np.abs(crackangle - np.pi/2.0) < np.pi/10 or np.abs(crackangle + np.pi/2.0) < np.pi/10:
        # approximately vertical

        int_centerpos = crackcenterx
        int_left=int_centerpos-integration_width/2.0
        int_right=int_centerpos+integration_width/2.0
        int_left_idx = (int_left-IniVal[0])/Step[0]
        int_right_idx = (int_right-IniVal[0])/Step[0]

        int_start_idx=min(int(round(int_left_idx)),int(round(int_right_idx)))
        int_start_idx=max(0,int_start_idx)
        int_end_idx=max(int(round(int_right_idx)),int(round(int_right_idx)))
        int_end_idx=min(int(DimLen[0])-1,int_end_idx)


        
        if cracktip1y < crackcentery:
            half_length = crackcentery-cracktip1y
            crack_region = (cracktip1y-extra_length/2.0,crackcentery+half_length+extra_length/2.0)
            pass
        else:
            half_length = cracktip1y-crackcentery
            crack_region = (crackcentery-half_length-extra_length/2.0,cracktip1y+extra_length/2.0)
            pass
        
        crack_region_idxs=np.array(((crack_region[0]-IniVal[1])/Step[1],(crack_region[1]-IniVal[1])/Step[1]),dtype='d')
        crack_region_start=min(int(round(crack_region_idxs[0])),int(round(crack_region_idxs[1])))  
        crack_region_start=max(0,crack_region_start)
        crack_region_end=max(int(round(crack_region_idxs[0])),int(round(crack_region_idxs[1])))
        crack_region_end=min(int(DimLen[1])-1,crack_region_end)

        use_data = evaluated.data[int_start_idx:(int_end_idx+1),crack_region_start:(crack_region_end+1),:]

        int_axis=0

        use_data_integrated = np.sum(use_data,axis=int_axis)*Step[0] # integral over dy

        crack_pos = bases[1][crack_region_start:(crack_region_end+1)]
        crack_length=abs(crackcentery-cracktip1y)*2.0
        int_pos=bases[0][int_start_idx:(int_end_idx+1)]

        crack_centerpos = crackcentery
        
        
        pass
    else:
        raise ValueError("Crack is neither approximately horizontal nor approximately vertical (angle = %f deg." % (crackangle*180.0/np.pi))


    n_r_bnds = 1+int(np.ceil(crack_length/(2.0*rstep)))
    r_bnds = np.arange(n_r_bnds,dtype='d')*rstep
    

    (bestfit,recon,s) = heatinvert(crack_pos,crack_centerpos,bases[2],(exc_t1+exc_t2)/2.0,(exc_t3+exc_t4)/2.0,matl_k,matl_rho*matl_c,r_bnds,1e-3,1.0,use_data_integrated,tikparam,ctx=ctx)

    
    # highres_idx identifies the first frame at least (400um)^2/alpha seconds beyond the average of exc_t1 and exc+t2
    highres_idx = np.where(bases[2] >= (exc_t1+exc_t2)/2.0 + (matl_rho*matl_c/matl_k)*(400e-6**2.0))[0][0]

    # excend_idx identifies the first frame at or beyond the average of exc_t3 and exc+t4
    excend_idx = np.where(bases[2] >= (exc_t3+exc_t4)/2.0)[0][0]

    
    # for highres, always use tikparam of 0.0...
    (bestfit_highres,recon_highres,s_highres) = heatinvert(crack_pos,crack_centerpos,bases[2][highres_idx:(highres_idx+1)],(exc_t1+exc_t2)/2.0,(exc_t3+exc_t4)/2.0,matl_k,matl_rho*matl_c,r_bnds,1e-3,1.0,use_data_integrated[:,highres_idx:(highres_idx+1)],0.0,ctx=ctx)

    return (crack_pos,
            crack_centerpos,
            int_pos,
            int_axis,
            bases[2], # timebase
            use_data,
            use_data_integrated,
            highres_idx,
            excend_idx,
            r_bnds,
            bestfit,recon,s,
            bestfit_highres,recon_highres,s_highres)
    

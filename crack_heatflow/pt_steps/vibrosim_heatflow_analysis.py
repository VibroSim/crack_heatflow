import sys
import os
import os.path

import numpy as np
import pandas as pd

from limatix.dc_value import hrefvalue as hrefv
from limatix.dc_value import numericunitsvalue as numericunitsv

from crack_heatflow import surface_heating

from matplotlib import pyplot as pl

def calc_heating_frame(along,across,unique_time,unique_r,r_inner,r_outer,timeseg_start,timeseg_end,alpha,k,side1_heating_reshape,side2_heating_reshape,frametime,ctx):
    recongrid = np.zeros((along.shape[0],across.shape[0]),dtype='d')

    if ctx is None:
        from crack_heatflow.heatpredict import surface_heating as surface_heating_unaccel
        surface_heating = lambda position_x,position_y,t,stripradius1,stripradius2,t1,t2,alpha,k,pos_side,ctx: surface_heating_unaccel(position_x,position_y,t,stripradius1,stripradius2,t1,t2,alpha,k,pos_side)
        pass
    else:
        from crack_heatflow.heatpredict_accel import surface_heating
        pass
    
    
    for timeidx in range(unique_time.shape[0]):
        recongrid += np.sum(surface_heating(along[:,np.newaxis,np.newaxis],across[np.newaxis,:,np.newaxis],frametime,r_inner[np.newaxis,np.newaxis,:],r_outer[np.newaxis,np.newaxis,:],timeseg_start[timeidx],timeseg_end[timeidx],alpha,k,False,ctx=ctx)*side1_heating_reshape[timeidx,np.newaxis,np.newaxis,:],axis=2) # sum over multiple radii
            
            
        recongrid += np.sum(surface_heating(along[:,np.newaxis,np.newaxis],across[np.newaxis,:,np.newaxis],frametime,r_inner[np.newaxis,np.newaxis,:],r_outer[np.newaxis,np.newaxis,:],timeseg_start[timeidx],timeseg_end[timeidx],alpha,k,True,ctx=ctx)*side2_heating_reshape[timeidx,np.newaxis,np.newaxis,:],axis=2) # sum over multiple radii
            
        pass
    
    return recongrid


def run(dc_dest_href,
        dc_measident_str,
        dc_heatingdata_href,
        dc_exc_t3_numericunits,
        dc_recon_size_across_numericunits,
        dc_recon_size_along_numericunits,
        dc_recon_stepsize_across_numericunits,
        dc_recon_stepsize_along_numericunits,
        dc_simulationcameranetd_numericunits,
        dc_spcThermalConductivity_numericunits,
        dc_spcSpecificHeatCapacity_numericunits,
        dc_Density_numericunits):

    k=dc_spcThermalConductivity_numericunits.value("W/m/K")
    c=dc_spcSpecificHeatCapacity_numericunits.value("J/kg/K")
    rho=dc_Density_numericunits.value("kg/m^3")
    
    size_along =  dc_recon_size_along_numericunits.value('m')
    size_across =  dc_recon_size_across_numericunits.value('m')
    step_along =  dc_recon_stepsize_along_numericunits.value('m')
    step_across =  dc_recon_stepsize_across_numericunits.value('m')

    along = np.arange(-size_along/2.0,size_along/2.0+step_along,step_along)
    across = np.arange(-size_across/2.0,size_across/2.0+step_across,step_across)
    (alonggrid,acrossgrid) = np.meshgrid(along,across,indexing="ij")
    
    heatingdata=pd.read_csv(dc_heatingdata_href.getpath(),sep="\t")
    time_key = '% t(s) '
    r_key =' r(m) '
    side1_heating_key=' side1_heating(W/m^2) '
    side2_heating_key=' side2_heating(W/m^2)'

    
    
    time=heatingdata[time_key].values

    r = heatingdata[r_key].values
    side1_heating = heatingdata[side1_heating_key]
    side2_heating = heatingdata[side2_heating_key]

    unique_time = np.unique(time)
    unique_r = np.unique(r)
    

    r_reshape = r.reshape(unique_time.shape[0],unique_r.shape[0])
    time_reshape = time.reshape(unique_time.shape[0],unique_r.shape[0])
    side1_heating_reshape = side1_heating.reshape(unique_time.shape[0],unique_r.shape[0])
    side2_heating_reshape = side2_heating.reshape(unique_time.shape[0],unique_r.shape[0])

    # Verify that the reshaped output is correctly a two-index dataset,
    # with indexes as specified by unique_time and unique_r
    assert(np.all(r_reshape==unique_r[np.newaxis,:]))
    assert(np.all(time_reshape==unique_time[:,np.newaxis]))
    
    dr = unique_r[1:]-unique_r[:-1]
    dr_full = np.concatenate((dr,np.array((dr[-1],))))

    r_inner = unique_r-dr_full/2.0
    r_outer = unique_r+dr_full/2.0
    r_inner[r_inner < 0.0]=0.0

    dt = unique_time[1:]-unique_time[:-1]
    dt_full = np.concatenate((dt,np.array((dt[-1],))))
    
    timeseg_start = unique_time-dt_full
    timeseg_end = unique_time+dt_full
    
    import pyopencl as cl 
    ctx = cl.create_some_context()  # set ctx equal to None in order to disable OpenCL acceleration
    
    surface_heating_t3 = calc_heating_frame(along,across,
                                            unique_time,unique_r,
                                            r_inner,r_outer,
                                            timeseg_start,timeseg_end,
                                            k/(rho*c),k,
                                            side1_heating_reshape,side2_heating_reshape,
                                            dc_exc_t3_numericunits.value("s"),ctx)


    camera_netd = dc_simulationcameranetd_numericunits.value("K")
    
    surface_heating_t3_noisy = surface_heating_t3 + np.random.randn(surface_heating_t3.shape)*camera_netd
    
    
    pl.figure()
    pl.imshow(surface_heating_t3.T,origin="lower",extent=((along[0]-step_along/2.0)*1e3,(along[-1]+step_along/2.0)*1e3,(across[0]-step_across/2.0)*1e3,(across[-1]+step_across/2.0)*1e3),cmap="hot")
    pl.xlabel('Position along crack (mm)')
    pl.xlabel('Position across crack (mm)')
    pl.title('t = %f s' % (dc_exc_t3_numericunits.value("s")))
    predicted_heating_plot_href = hrefv(quote(dc_measident_str+"_predicted_heating_plot.png"),dc_dest_href)
    pl.savefig(predicted_heating_plot_href.getpath(),dpi=300)
    

    pl.figure()
    pl.imshow(surface_heating_t3_noisy.T,origin="lower",extent=((along[0]-step_along/2.0)*1e3,(along[-1]+step_along/2.0)*1e3,(across[0]-step_across/2.0)*1e3,(across[-1]+step_across/2.0)*1e3),cmap="hot",vmin=-camera_netd,vmax=camera_netd*9)
    pl.xlabel('Position along crack (mm)')
    pl.xlabel('Position across crack (mm)')
    pl.title('t = %f s' % (dc_exc_t3_numericunits.value("s")))
    noisy_predicted_heating_plot_href = hrefv(quote(dc_measident_str+"_noisy_predicted_heating_plot.png"),dc_dest_href)
    pl.savefig(noisy_predicted_heating_plot_href.getpath(),dpi=300)
    

    return {
        "dc:predicted_heating_plot": predicted_heating_plot_href,
        "dc:noisy_predicted_heating_plot": noisy_predicted_heating_plot_href,
    }


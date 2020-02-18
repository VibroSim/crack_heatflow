import sys
import os
import os.path

try:
    # py2.x
    from urllib import pathname2url
    from urllib import url2pathname
    from urllib import quote
    from urllib import unquote
    pass
except ImportError:
    # py3.x
    from urllib.request import pathname2url
    from urllib.request import url2pathname
    from urllib.parse import quote
    from urllib.parse import unquote
    pass


import numpy as np
import pandas as pd
import scipy
import scipy.interpolate

from limatix.dc_value import hrefvalue as hrefv
from limatix.dc_value import numericunitsvalue as numericunitsv

from crack_heatflow import surface_heating

from matplotlib import pyplot as pl

def calc_heating_finitedifference(z,z_bnd,dz,along,along_bnd,step_along,across,across_bnd,step_across,unique_time,dt_full,unique_r,r_inner,r_outer,k,rho,c,side1_heating_reshape,side2_heating_reshape,max_timestep,time_limit):
    import heatsim2

    dt = (unique_time[-1]-unique_time[0])/(unique_time.shape[0]-1)
    upsamplefactor=1
    nt = unique_time.shape[0]

    t_bnd_orig = (unique_time[0]-dt/2.0) + np.arange(nt+1,dtype='d')*dt

    while dt > max_timestep:
        upsamplefactor *= 2 
        dt /= 2.0
        nt *= upsamplefactor
        pass

    # adjust nt according to time_limit
    
    nt = int(np.ceil((time_limit-unique_time[0])/dt + 0.5))
    
    t_bnd_input = (unique_time[0]-dt/2.0) + np.arange(nt+1,dtype='d')*dt
    t_bnd_output = (unique_time[0]) + np.arange(nt+1,dtype='d')*dt

    (z_bnd_z,z_bnd_across,z_bnd_along)=np.meshgrid(z_bnd,across,along,indexing='ij')

    (across_bnd_z,across_bnd_across,across_bnd_along)=np.meshgrid(z,across_bnd,along,indexing='ij')
    
    (along_bnd_z,along_bnd_across,along_bnd_along)=np.meshgrid(z,across,along_bnd,indexing='ij')


    (zgrid,acrossgrid,alonggrid) = np.meshgrid(z,across,along,indexing="ij")

    materials=(
        # material 0: specimen
        (heatsim2.TEMPERATURE_COMPUTE,k,rho,c),
    )
    boundaries=(
        # boundary 0: conducting
        (heatsim2.boundary_conducting,),
        (heatsim2.boundary_insulating,),
    )
    volumetric=(  # on material grid
        # 0: nothing
        (heatsim2.NO_SOURCE,),
        #1: stepped source 
        [heatsim2.IMPULSE_SOURCE,0.0, [] ], # t0 (sec), Energy (J/m^3) as a list so we can monkey-patch them during iteration
    )
    
    # initialize all elements to zero
    (material_elements,
     boundary_z_elements,
     boundary_across_elements,
     boundary_along_elements,
     volumetric_elements)=heatsim2.zero_elements(z.shape[0],across.shape[0],along.shape[0]) 

    # set x and y and z=0 edges to insulating
    boundary_along_elements[:,:,0]=1 # insulating
    boundary_along_elements[:,:,-1]=1 # insulating
    boundary_across_elements[:,0,:]=1 # insulating
    boundary_across_elements[:,-1,:]=1 # insulating
    boundary_z_elements[0,:,:]=1 # insulating
    boundary_z_elements[-1,:,:]=1 # insulating
    
    # Source turned on for across=0 plane
    volumetric_elements[(abs(acrossgrid) < 1e-6)]=1  # impulse source
    
    
    (ADI_params,ADI_steps)=heatsim2.setup(z[0],across[0],along[0],
                                          dz,step_across,step_along,
                                          z.shape[0],across.shape[0],along.shape[0],
                                          dt,
                                          materials,
                                          boundaries,
                                          volumetric,
                                      material_elements,
                                      boundary_z_elements,
                                      boundary_across_elements,
                                      boundary_along_elements,
                                      volumetric_elements)

    print((t_bnd_input.shape[0]-1,z.shape[0],across.shape[0],along.shape[0]))

    T=np.zeros((t_bnd_input.shape[0]-1,z.shape[0],across.shape[0],along.shape[0]),dtype='f')

    last_temp = np.zeros((z.shape[0],across.shape[0],along.shape[0]),dtype='d') # initial condition
    for tidx in range(t_bnd_input.shape[0]-1):
        print("tidx=%d/%d" % (tidx,t_bnd_input.shape[0]-1))
        t_start_input=t_bnd_input[tidx] # our time is centered over the input sample
        t_end_input=t_bnd_input[tidx+1]
        t_center_input=(t_start_input+t_end_input)/2.0
        
        t_start_output=t_bnd_output[tidx] # our result is centered over the output sample, which is 1/2 dt delayed
        t_end_output=t_bnd_output[tidx+1]
        t_center_output=(t_start_output+t_end_output)/2.0

        # assign volumetric heating
        volumetric[1][1] = t_center_input # force a match for this time step
        volumetric_r = (np.sqrt(zgrid**2.0 + alonggrid**2.0))[volumetric_elements==1]
        volumetric_sourceintensity=np.zeros(volumetric_r.shape,dtype='d')


        if tidx//upsamplefactor < unique_time.shape[0]:
            volumetric_alongpos = (alonggrid > 0.0)[volumetric_elements==1] # are we on the positive side of the along axis? (i.e. side #2) 

        
            tck_side1 = scipy.interpolate.splrep(unique_r,side1_heating_reshape[tidx//upsamplefactor,:],k=1,task=0,s=0.0)

            tck_side2 = scipy.interpolate.splrep(unique_r,side2_heating_reshape[tidx//upsamplefactor,:],k=1,task=0,s=0.0)
            
            #volumetric_sourceintensity[~volumetric_alongpos] = scipy.interpolate.splev(heating_r_side1,tck_side1,ext=0)  # This is what we are doing, but it doesn't work quite right because of extrapolation at positive r. 

            heating_r_side1 = volumetric_r[~volumetric_alongpos]
            heating_intensity_side1 = np.zeros(heating_r_side1.shape,dtype='d')
            
            heating_intensity_side1[heating_r_side1 < unique_r[-1]] = scipy.interpolate.splev(heating_r_side1[heating_r_side1 < unique_r[-1]],tck_side1,ext=0)
            
            volumetric_sourceintensity[~volumetric_alongpos] = heating_intensity_side1

            
            #volumetric_sourceintensity[volumetric_alongpos] = scipy.interpolate.splev(volumetric_r[volumetric_alongpos],tck_side2,ext=0) # This is what we are doing, but it doesn't work quite right because of extrapolation at positive r. 


            heating_r_side2 = volumetric_r[volumetric_alongpos]
            heating_intensity_side2 = np.zeros(heating_r_side2.shape,dtype='d')
            
            heating_intensity_side2[heating_r_side2 < unique_r[-1]] = scipy.interpolate.splev(heating_r_side2[heating_r_side2 < unique_r[-1]],tck_side2,ext=0)
            
            volumetric_sourceintensity[volumetric_alongpos] = heating_intensity_side2


            pass
        
        volumetric[1][2] = volumetric_sourceintensity*dt/step_across  # dt/step_across converts from W/m^2 to J/m^3 volumetric source 
        
        last_temp = heatsim2.run_adi_steps(ADI_params,ADI_steps,t_center_input,dt,last_temp,volumetric_elements,volumetric)

        T[tidx,:,:,:] = last_temp
        pass
    return (t_bnd_output,T)
        
    
def calc_heating_integral(along,across,unique_time,unique_r,r_inner,r_outer,timeseg_start,timeseg_end,alpha,k,side1_heating_reshape,side2_heating_reshape,frametime,ctx):
    recongrid = np.zeros((along.shape[0],across.shape[0]),dtype='d')

    if ctx is None:
        from crack_heatflow.heatpredict import surface_heating as surface_heating_unaccel
        surface_heating = lambda position_x,position_y,t,stripradius1,stripradius2,t1,t2,alpha,k,pos_side,ctx: surface_heating_unaccel(position_x,position_y,t,stripradius1,stripradius2,t1,t2,alpha,k,pos_side)
        pass
    else:
        from crack_heatflow.heatpredict_accel import surface_heating
        pass
    
    
    for timeidx in range(unique_time.shape[0]):
        print("timeidx=%d/%d" % (timeidx,unique_time.shape[0]))
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
        dc_Density_numericunits,
        dc_heatflow_method_str="finitedifference",
        dc_heatflow_fd_thick_numericunits=numericunitsv(5e-3,'m'),
        dc_heatflow_max_timestep_numericunits=numericunitsv(10e-3,'s')):

    k=dc_spcThermalConductivity_numericunits.value("W/m/K")
    c=dc_spcSpecificHeatCapacity_numericunits.value("J/kg/K")
    rho=dc_Density_numericunits.value("kg/m^3")

        
    heatingdata=pd.read_csv(dc_heatingdata_href.getpath(),sep="\t")
    time_key = '% t(s) '
    r_key =' r(m) '
    side1_heating_key=' side1_heating(W/m^2) '
    side2_heating_key=' side2_heating(W/m^2)'

    
    
    time=heatingdata[time_key].values

    r = heatingdata[r_key].values
    side1_heating = heatingdata[side1_heating_key].values
    side2_heating = heatingdata[side2_heating_key].values

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
    dr_typ = np.median(dr)
    
    r_inner = unique_r-dr_full/2.0
    r_outer = unique_r+dr_full/2.0
    r_inner[r_inner < 0.0]=0.0

    dt = unique_time[1:]-unique_time[:-1]
    dt_full = np.concatenate((dt,np.array((dt[-1],))))
    
    timeseg_start = unique_time-dt_full/2.0
    timeseg_end = unique_time+dt_full/2.0


    size_along =  dc_recon_size_along_numericunits.value('m')
    size_across =  dc_recon_size_across_numericunits.value('m')
    step_along =  dc_recon_stepsize_along_numericunits.value('m')
    step_across =  dc_recon_stepsize_across_numericunits.value('m')



    # Step sizes should be smaller than 2*dr
    while step_along > dr_typ*2.0:
        step_along /= 2
        pass

    while step_across > dr_typ*2.0:
        step_across /= 2
        pass

    
    n_along = int(round(size_along/step_along/2.0))*2  # n_along must be even
    n_across = int(round(size_across/step_across/2.0+1.0))*2 - 1 # n_across must be odd so that there is one centered at zero

    along = (np.arange(n_along,dtype='d')-n_along/2.0 + 1)*step_along
    along_bnd = (np.arange(n_along+1,dtype='d')-n_along/2.0 + 0.5)*step_along
    across = (np.arange(n_across,dtype='d') - (n_across-1)/2.0)*step_across
    across_bnd = (np.arange(n_across+1,dtype='d')-(n_across)/2.0)*step_across

    if dc_heatflow_method_str=="finitedifference":
        dz = min(step_along,step_across)
        z_bnd = np.arange(0,dc_heatflow_fd_thick_numericunits.value('m')+dz,dz)
        z = (z_bnd[:-1]+z_bnd[1:])/2.0
        
        pass
    
    
    #(alonggrid,acrossgrid) = np.meshgrid(along,across,indexing="ij")

    if dc_heatflow_method_str=="finitedifference":

        #raise ValueError("Debug")

        (t_bnd_output,T) = calc_heating_finitedifference(z,z_bnd,dz,along,along_bnd,step_along,across,across_bnd,step_across,unique_time,dt_full,unique_r,r_inner,r_outer,k,rho,c,side1_heating_reshape,side2_heating_reshape,dc_heatflow_max_timestep_numericunits.value("s"),dc_exc_t3_numericunits.value("s"))
        t_center_output = (t_bnd_output[:-1]+t_bnd_output[1:])/2.0
        t_extract_idx = np.argmin(abs(dc_exc_t3_numericunits.value("s")-t_center_output))
        surface_heating_t3 = T[t_extract_idx,0,:,:].T
        pass
    elif dc_heatflow_method_str=="greensintegration":
        import pyopencl as cl 
        ctx = cl.create_some_context()  # set ctx equal to None in order to disable OpenCL acceleration
        
        surface_heating_t3 = calc_heating_integral(along,across,
                                                   unique_time,unique_r,
                                                   r_inner,r_outer,
                                                   timeseg_start,timeseg_end,
                                                   k/(rho*c),k,
                                                   side1_heating_reshape,side2_heating_reshape,
                                                   dc_exc_t3_numericunits.value("s"),ctx)
        pass
    
    camera_netd = dc_simulationcameranetd_numericunits.value("K")
    
    surface_heating_t3_noisy = surface_heating_t3 + np.random.randn(*surface_heating_t3.shape)*camera_netd
    
    
    pl.figure()
    pl.imshow(surface_heating_t3.T,origin="lower",
              extent=((along[0]-step_along/2.0)*1e3,
                      (along[-1]+step_along/2.0)*1e3,
                      (across[0]-step_across/2.0)*1e3,
                      (across[-1]+step_across/2.0)*1e3),
              cmap="hot")
    pl.colorbar()
    pl.xlabel('Position along crack (mm)')
    pl.ylabel('Position across crack (mm)')
    pl.title('t = %f s' % (dc_exc_t3_numericunits.value("s")))
    heating_predicted_plot_href = hrefv(quote(dc_measident_str+"_heating_predicted_plot.png"),dc_dest_href)
    pl.savefig(heating_predicted_plot_href.getpath(),dpi=300)
    

    pl.figure()
    pl.imshow(surface_heating_t3_noisy.T,origin="lower",
              extent=((along[0]-step_along/2.0)*1e3,
                      (along[-1]+step_along/2.0)*1e3,
                      (across[0]-step_across/2.0)*1e3,
                      (across[-1]+step_across/2.0)*1e3),
              cmap="hot",vmin=-camera_netd,vmax=camera_netd*9)
    pl.colorbar()
    pl.xlabel('Position along crack (mm)')
    pl.ylabel('Position across crack (mm)')
    pl.title('t = %f s' % (dc_exc_t3_numericunits.value("s")))
    heating_predicted_plot_noisy_href = hrefv(quote(dc_measident_str+"_heating_predicted_plot_noisy.png"),dc_dest_href)
    pl.savefig(heating_predicted_plot_noisy_href.getpath(),dpi=300)
    

    return {
        "dc:heating_predicted_plot": heating_predicted_plot_href,
        "dc:heating_predicted_plot_noisy": heating_predicted_plot_noisy_href,
    }


import numpy as np
from matplotlib import pyplot as pl

import heatsim2

import pyopencl as cl

from crackheat_inversion.heatinversion import heatinvert
from crackheat_inversion import surface_heating_y_integral

#from function_as_script import scriptify
#from crackheat_inversion.heatinvert import heatinvert as heatinvert_function
#heatinvert = scriptify(heatinvert_function)

ctx=cl.create_some_context()

k=6.7 # rough thermal conductivity of titanium, W/(m*deg K)
rho=4430.0 # rough density for titanium
c=526.0 # rough specific heat for titanium, J/kg*k

t1=0.2  # excitation start
t2=1.2  # excitation end

xcenter=3e-3

grid_refinement=1

# Create x,y,z voxel center coords
nz=30*grid_refinement
ny=20*grid_refinement+1
nx=30*grid_refinement+1
(dz,dy,dx,
 z,y,x,
 zgrid,ygrid,xgrid,
 z_bnd,y_bnd,x_bnd,
 z_bnd_z,z_bnd_y,z_bnd_x,
 y_bnd_z,y_bnd_y,y_bnd_x,
 x_bnd_z,x_bnd_y,x_bnd_x,
 r3d,r2d) = heatsim2.build_grid(0,10e-3,nz,
                                -7e-3,7e-3,ny,
                                -10e-3,10e-3,nx)

materials=(
    # material 0: titanium
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
    (heatsim2.STEPPED_SOURCE,t1,t2,1000.0/dy), # t0 (sec), t1 (sec), Power W/m^2/dy 
)

# initialize all elements to zero
(material_elements,
 boundary_z_elements,
 boundary_y_elements,
 boundary_x_elements,
 volumetric_elements)=heatsim2.zero_elements(nz,ny,nx) 

# set x and y and z=0 edges to insulating
boundary_x_elements[:,:,0]=1 # insulating
boundary_x_elements[:,:,-1]=1 # insulating
boundary_y_elements[:,0,:]=1 # insulating
boundary_y_elements[:,-1,:]=1 # insulating
boundary_z_elements[0,:,:]=1 # insulating
boundary_z_elements[-1,:,:]=1 # insulating


# Source turned on between 1mm and 3mm
volumetric_elements[(abs(ygrid) < 1e-6) & (np.sqrt((xgrid-xcenter)**2+zgrid**2) >= 1e-3) & (np.sqrt((xgrid-xcenter)**2+zgrid**2) < 3e-3)]=1  # stepped source


t0=0
dt=.01/grid_refinement
nt_bnd=200*grid_refinement
nt_centers=nt_bnd-1


t_bnd=t0+np.arange(nt_bnd,dtype='d')*dt
t_centers=(t_bnd[:-1]+t_bnd[1:])/2.0

# NOTE: Important that t1 and t2 line up with elements of t_bnd
t1idx=np.argmin(abs(t1-t_bnd))
t2idx=np.argmin(abs(t2-t_bnd))
assert(t_bnd[t1idx] <= t1 and abs(t1-t_bnd[t1idx]) < 1e-4)
assert(t_bnd[t2idx] <= t2 and abs(t2-t_bnd[t2idx]) < 1e-4)


(ADI_params,ADI_steps)=heatsim2.setup(z[0],y[0],x[0],
                                      dz,dy,dx,
                                      nz,ny,nx,
                                      dt,
                                      materials,
                                      boundaries,
                                      volumetric,
                                      material_elements,
                                      boundary_z_elements,
                                      boundary_y_elements,
                                      boundary_x_elements,
                                      volumetric_elements)


T=np.zeros((nt_centers,nz,ny,nx),dtype='d')
last_temp=np.zeros((nz,ny,nx),dtype='d') # initial condition
for tcnt in range(nt_centers):
    tval_center=t_centers[tcnt]
    tval_bnd=t_bnd[tcnt]
    print "tval_center=%f" % (tval_center)
    T[tcnt,...]=heatsim2.run_adi_steps(ADI_params,ADI_steps,tval_bnd,dt,last_temp,volumetric_elements,volumetric)
    last_temp=T[tcnt,...]
    pass

                    
# Evaluate at z=0, integrate over y, transpose so nx by nt
integrated = T[:,0,:,:].sum(1).transpose()*dy

r_bnds = np.arange(0,6e-3,1e-3)
r_centers=(r_bnds[:-1]+r_bnds[1:])/2.0

# Low resolution, using all frames
tikparam=2e-7
(bestfit,recon,s) = heatinvert(x,xcenter,t_centers,t1,t2,k,rho*c,r_bnds,1e-3,1.0,integrated,tikparam)

# high resolution, using only one frame
frameno = t1idx+10*grid_refinement
tikparam_highres= 0.0
(bestfit_highres,recon_highres,s_highres) = heatinvert(x,xcenter,t_centers[frameno:(frameno+1)],t1,t2,k,rho*c,r_bnds,1e-3,1.0,integrated[:,frameno:(frameno+1)],tikparam_highres,ctx=ctx)


# Forward prediction from lowres/highres:
predict_lowres=np.zeros((x.shape[0],t_centers.shape[0]),dtype='d')
predict_highres=np.zeros((x.shape[0],t_centers.shape[0]),dtype='d')
for r_idx in range(r_bnds.shape[0]-1):
    halfsemi_neg_heating = surface_heating_y_integral(1e-3,1.0,x[:,np.newaxis]-xcenter,t_centers[np.newaxis,:],r_bnds[r_idx],r_bnds[r_idx+1],t1,t2,k/(rho*c),k,False,ctx=ctx)
    halfsemi_pos_heating = surface_heating_y_integral(1e-3,1.0,x[:,np.newaxis]-xcenter,t_centers[np.newaxis,:],r_bnds[r_idx],r_bnds[r_idx+1],t1,t2,k/(rho*c),k,True,ctx=ctx)

    predict_lowres += halfsemi_neg_heating*bestfit[r_idx*2]
    predict_lowres += halfsemi_pos_heating*bestfit[r_idx*2+1]

    predict_highres += halfsemi_neg_heating*bestfit_highres[r_idx*2]
    predict_highres += halfsemi_pos_heating*bestfit_highres[r_idx*2+1]

    pass


pl.figure()
pl.subplot(2,1,1)
pl.imshow(integrated,vmin=0,vmax=np.max(integrated)*1.1)
pl.colorbar()
pl.title('Forward finite difference sim (lowres)')

pl.subplot(2,1,2)
pl.imshow(recon,vmin=0,vmax=np.max(integrated)*1.1)
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
pl.plot(integrated[:,frameno:(frameno+1)])
pl.plot(recon_highres[:,0])
pl.title('Forward finite difference sim and reconstruction (highres)')
pl.legend(('Finite difference','Reconstruction'))

pl.figure()
pl.plot(r_centers,bestfit_highres[::2],'-',
        r_centers,bestfit_highres[1::2],'-')
pl.title('recovered source intensity as function of r (highres)')
pl.ylabel('Source intensity, W/m^2')
pl.legend(('x < 0','x > 0'))

pl.figure()
pl.plot(s_highres)
pl.plot((0,s_highres.shape[0]-1),(tikparam_highres,tikparam_highres))
pl.title('Tikhonov parameter diagnostic (highres)')
pl.legend(('Singular values','Tikhonov parameter'))

pl.figure()
pl.imshow(predict_lowres)
pl.colorbar()

pl.figure()
pl.imshow(predict_highres)
pl.colorbar()

pl.figure()
pl.plot(integrated[:,frameno:(frameno+1)])
pl.plot(predict_lowres[:,frameno:(frameno+1)])
pl.plot(predict_highres[:,frameno:(frameno+1)])
pl.plot(recon_highres[:,0])
pl.legend(('original sim','predict_lowres','predict_highres','recon_highres'))

pl.figure()
pl.plot(integrated[:,t2idx:(t2idx+1)])
pl.plot(predict_lowres[:,t2idx:(t2idx+1)])
pl.plot(predict_highres[:,t2idx:(t2idx+1)])
#pl.plot(recon_highres[:,0])
pl.legend(('original sim','predict_lowres','predict_highres'))

pl.show()

import numpy as np
from matplotlib import pyplot as pl

import heatsim2

import pyopencl as cl

from crack_heatflow import surface_heating

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

Power_watts_per_m2 = 1000.0

volumetric=(  # on material grid
    # 0: nothing
    (heatsim2.NO_SOURCE,),
    #1: stepped source 
    (heatsim2.STEPPED_SOURCE,t1,t2,Power_watts_per_m2/dy), # t0 (sec), t1 (sec), Power W/m^2/dy 
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


y0idx = np.where(abs(y) < 1e-6)[0][0]

t0=0
dt=.01/grid_refinement
nt_bnd=200*grid_refinement
nt_centers=nt_bnd-1


t_bnd=t0+np.arange(nt_bnd,dtype='d')*dt
t_centers=(t_bnd[:-1]+t_bnd[1:])/2.0

# NOTE: Important that t1 and t2 line up with elements of t_bnd
t1idx=np.argmin(abs(t1-t_bnd))
t2idx=np.where((t_bnd-t2) < 0)[0][-1]
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
    print("tval_center=%f" % (tval_center))
    T[tcnt,...]=heatsim2.run_adi_steps(ADI_params,ADI_steps,tval_bnd,dt,last_temp,volumetric_elements,volumetric)
    last_temp=T[tcnt,...]
    pass

                    
## Evaluate at z=0, integrate over y, transpose so nx by nt
#integrated = T[:,0,:,:].sum(1).transpose()*dy

r_bnds = np.arange(0,6e-3,1e-3)
r_centers=(r_bnds[:-1]+r_bnds[1:])/2.0

# Forward prediction
predict=np.zeros((t_centers.shape[0],y.shape[0],x.shape[0]),dtype='d')

# Integrate over heating semicircles from 1mm to 3mm
halfsemi_neg_heating = surface_heating(x[np.newaxis,np.newaxis,:]-xcenter,y[np.newaxis,:,np.newaxis],t_centers[:,np.newaxis,np.newaxis],1e-3,3e-3,t1,t2,k/(rho*c),k,False,ctx=ctx)
halfsemi_pos_heating = surface_heating(x[np.newaxis,np.newaxis,:]-xcenter,y[np.newaxis,:,np.newaxis],t_centers[:,np.newaxis,np.newaxis],1e-3,3e-3,t1,t2,k/(rho*c),k,True,ctx=ctx)

predict += halfsemi_neg_heating * (Power_watts_per_m2)
predict += halfsemi_pos_heating * (Power_watts_per_m2)


frameno = t1idx+10*grid_refinement

pl.figure()
pl.imshow(T[frameno,0,:,:],vmin=0,vmax=np.max(T[frameno,0,:,:])*1.1)
pl.colorbar()
pl.title('Forward finite difference sim; t=%f s' % (t_centers[frameno]))


pl.figure()
pl.imshow(predict[frameno,:,:],vmin=0,vmax=np.max(T[frameno,0,:,:])*1.1)
pl.colorbar()
pl.title("Direct integration; t=%f s" % (t_centers[frameno]))

pl.figure()
pl.imshow(T[t2idx,0,:,:],vmin=0,vmax=np.max(T[t2idx,0,:,:])*1.1)
pl.colorbar()
pl.title('Forward finite difference sim; t=%f s' % (t_centers[t2idx]))


pl.figure()
pl.imshow(predict[t2idx,:,:],vmin=0,vmax=np.max(T[t2idx,0,:,:])*1.1)
pl.colorbar()
pl.title("Direct integration; t=%f s" % (t_centers[t2idx]))

pl.figure()
pl.plot(x,T[frameno,0,y0idx,:])
pl.plot(x,predict[frameno,y0idx,:])
pl.legend(('Finite difference','Direct integration'))
pl.title("t=%f s" % (t_centers[frameno]))

pl.figure()
pl.plot(x,T[t2idx,0,y0idx,:])
pl.plot(x,predict[t2idx,y0idx,:])
pl.legend(('Finite difference','Direct integration'))
pl.title("t=%f s" % (t_centers[t2idx]))

pl.show()

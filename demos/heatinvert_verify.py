import numpy as np
from matplotlib import pyplot as pl

import heatsim2
from function_as_script import scriptify
from heatinvert import heatinvert as heatinvert_function

from heatpredict_accel import surface_heating_y_integral
import pyopencl as cl

heatinvert = scriptify(heatinvert_function)


k=22.4 # rough thermal conductivity of titanium, W/(m*deg K)
rho=4430.0 # rough density for titanium
c=540.0 # rough specific heat for titanium, J/kg*k

t1=0.2  # excitation start
t2=1.2  # excitation end

xcenter=3e-3

# Create x,y,z voxel center coords
nz=30
ny=21
nx=31
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


# Source turned on between 0mm and 3mm
volumetric_elements[(abs(ygrid) < 1e-6) & (np.sqrt((xgrid-xcenter)**2+zgrid**2) >= 0e-3) & (np.sqrt((xgrid-xcenter)**2+zgrid**2) < 3e-3)]=1  # stepped source


t0=0
dt=.01
nt=200

tvec=t0+np.arange(nt,dtype='d')*dt

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


T=np.zeros((nt,nz,ny,nx),dtype='d')
for tcnt in range(nt-1):
    t=t0+dt*tcnt
    print "t=%f" % (t)
    T[tcnt+1,::]=heatsim2.run_adi_steps(ADI_params,ADI_steps,t,dt,T[tcnt,::],volumetric_elements,volumetric)
    pass

                    
# Evaluate at z=0, integrate over y, transpose so nx by nt
integrated = T[:,0,:,:].sum(1).transpose()*dy

r_bnds = np.arange(0,6e-3,1e-3)
r_centers=(r_bnds[:-1]+r_bnds[1:])/2.0

(bestfit,recon,s) = heatinvert(x,xcenter,tvec,t1,t2,k,rho*c,r_bnds,1e-3,1.0,integrated,2e-7)

pl.figure()
pl.subplot(2,1,1)
pl.imshow(integrated,vmin=0,vmax=np.max(integrated)*1.1)
pl.colorbar()
pl.title('Forward finite difference sim')

pl.subplot(2,1,2)
pl.imshow(recon,vmin=0,vmax=np.max(integrated)*1.1)
pl.colorbar()
pl.title('Inversion reconstruction')

pl.figure()
pl.plot(r_centers,bestfit[::2],'-',
        r_centers,bestfit[1::2],'-')
pl.title('recovered source intensity as function of r')
pl.ylabel('Source intensity, W/m^2')
pl.legend(('x < 0','x > 0'))

pl.figure()
pl.plot(s)
pl.plot((0,s.shape[0]-1),(tikparam,tikparam))
pl.title('Tikhonov parameter diagnostic')
pl.legend(('Singular values','Tikhonov parameter'))


pl.show()

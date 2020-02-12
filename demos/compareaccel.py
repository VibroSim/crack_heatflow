import sys
import os
import numpy as np
from matplotlib import pyplot as pl

from crack_heatflow.heatpredict import surface_heating
from crack_heatflow.heatpredict_accel import surface_heating as surface_heating_accel


if __name__=="__main__":
    x=np.linspace(-5.0e-3,5.0e-3,20)
    y=np.linspace(-4.0e-3,4.0e-3,16)
    t=np.linspace(0.0,4.0,10)
    
    t1=0.2
    t2=1.2
    k=22.4 # rough thermal conductivity of titanium, W/(m*deg K)
    rho=4430.0 # rough density for titanium
    c=540.0 # rough specific heat for titanium, J/kg*k
    alpha=k/(rho*c)  # m^2/s 
    
    stripradius1=2e-3
    stripradius2=2.1e-3

    posside=False
    
    (x_nd,y_nd,t_nd)=np.meshgrid(x,y,t,indexing="ij")
    
    T_nd = surface_heating(x_nd,y_nd,t_nd,stripradius1,stripradius2,t1,t2,alpha,k,posside)

    T_nd_accel = surface_heating_accel(x_nd,y_nd,t_nd,stripradius1,stripradius2,t1,t2,alpha,k,posside)

    error = T_nd-T_nd_accel
    error_norm = np.linalg.norm(error.ravel())
    signal_norm=np.linalg.norm(T_nd.ravel())
    
    
    pl.figure()
    pl.imshow(T_nd[:,:,2])
    pl.colorbar()
    pl.title('Python/CPU implementation')
    
    pl.figure()
    pl.imshow(T_nd_accel[:,:,2])
    pl.colorbar()
    pl.title('GPU implementation')

    pl.figure()
    pl.imshow(error[:,:,2])
    pl.colorbar()
    pl.title('Error residual; relative error = %g' % (error_norm/signal_norm))

    assert(error_norm/signal_norm < 1e-4)
    
    pl.show()

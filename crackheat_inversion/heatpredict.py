
from scipy.special import erfc
import scipy
import scipy.integrate
import numpy as np

# By greensfcn_doc.pdf:
# The Green's function for a point source in a half-space is:
# (2/(rho*c*(4*pi*(k/(rho*c))*t)^(3/2)))*exp(-r^2/(4*k*t/(rho*c)))
# for a unit energy (1 J) instantaneous point heat source at t=0
# The integral over time of this is:
#  (by Wolfram Alpha "integral of (2/(rho*c*(4*pi*(k/(rho*c))*t)^(3/2)))*exp(-r^2/(4*k*t/(rho*c))) dt"):
# and converting erf() to 1-erfc()..., assuming c,k,r,t, and rho positive
#   erfc(r/sqrt(4*(k/(rho*c))*t)) / (2*pi*k*r)
# In these calculations we leave the leading coefficient of
# 1/(2*pi*k) to the end.


# !!! Need to check leading coefficient!!!
def Tcrack(R,t,t1,t2,alpha):
    factor = 1.0/(R)
    term1=0.0

    if not isinstance(t,np.ndarray) and t <= t1:
        term1=0.0
        pass
    elif not isinstance(t,np.ndarray) and t > t1:
        term1=erfc(R/np.sqrt(4.0*alpha*(t-t1)))
    else :
        (R_bc,t_bc)=np.broadcast_arrays(R,t)
        
        term1=np.zeros(R_bc.shape,dtype='d')
        
        
        term1[t_bc > t1] = erfc(R_bc[t_bc > t1]/np.sqrt(4.0*alpha*(t_bc[t_bc > t1]-t1)))

        pass
        
    
    if not isinstance(t,np.ndarray) and t <= t2:
        term2=0.0
        pass
    elif not isinstance(t,np.ndarray) and t > t2:
        term2=-erfc(R/np.sqrt(4.0*alpha*(t-t2)))
    else :
        term2=np.zeros(term1.shape,dtype='d')

        t_bc=np.broadcast_to(t,term1.shape)
        R_bc=np.broadcast_to(R,term1.shape)
        
        term2[t_bc > t2] = -erfc(R_bc[t_bc > t2]/np.sqrt(4.0*alpha*(t_bc[t_bc > t2]-t2)))                
        pass

    
    return factor*(term1+term2)

def surface_heating_nonvect(position_x,position_y,t,stripradius1,stripradius2,t1,t2,alpha,k,pos_side):
    """Given a position (x,y) on the surface
       and a crackcenter (x,y) on the surface, 
       crack oriented in the x-z plane,
       and a heating semicircle of inner radius stripradius1
       and outer radius stripradius2
       calculate the heating at a particular time t relative 
       to excitation starting at zero, with excitation length t2"""
    
    crackcenter=(0.0,0.0)

    assert(not isinstance(t,np.ndarray))
    
    heating_rtheta_times_r = lambda R,theta: Tcrack( np.sqrt((crackcenter[0]+R*np.sin(theta) - position_x)**2.0 + (crackcenter[1]-position_y)**2.0 + (R*np.cos(theta))**2.0),t,t1,t2,alpha)*R

    heating_theta = lambda theta: scipy.integrate.quad(heating_rtheta_times_r,stripradius1,stripradius2,(theta,))[0]

    if (pos_side):
        heating = scipy.integrate.quad(heating_theta,0.0,np.pi/2.0)[0]
        pass
    else:
        heating = scipy.integrate.quad(heating_theta,-np.pi/2.0,0.0)[0]
        pass
    
    return heating/(2.0*np.pi*k)  # multiply by leading factor of 1.0/(2.0*pi*k).. see comment at top of file




surface_heating=np.vectorize(surface_heating_nonvect,('d'),excluded=(5,6,7,8,9))

if __name__ == "__main__":

    from matplotlib import pyplot as pl

    k=22.4 # rough thermal conductivity of titanium, W/(m*deg K)
    rho=4430.0 # rough density for titanium
    c=540.0 # rough specific heat for titanium, J/kg*k
    alpha=k/(rho*c)  # m^2/s 
    stripradius1=2e-3
    stripradius2=2.1e-3
    t1=0.2
    t2=1.2
    
    x=np.linspace(-5.0e-3,5.0e-3,20)
    y=np.linspace(-4.0e-3,4.0e-3,16)
    t=np.linspace(0.0,4.0,10)
    (x_nd,y_nd,t_nd)=np.meshgrid(x,y,t,indexing="ij")
    
    surface_heating_vals=surface_heating(x_nd,y_nd,t_nd,stripradius1,stripradius2,t1,t2,alpha,True)
    
    pass

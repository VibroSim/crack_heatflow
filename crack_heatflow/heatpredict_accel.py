import sys
import os
import os.path
import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as pl
import scipy.integrate
from scipy.integrate import quad
import time

try:
    import pyopencl as cl
    pass
except ImportError:
    cl=None
    pass

    
# regenerate qagse_fparams.c with:
# f2c -a qagse_fparams.f
# patch -p0 <qagse_fparams.patch

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


# NOTE: kernelpattern is processed with
# '%' (printf) operator, so
# be careful about percent signs in it...
# '%s's replaced by contents of qagse_fparams.c
kernelpattern=r"""

#ifdef static 
#undef static
#endif

#define static const __constant // f2c generates static when it means const


typedef float doublereal;  // perform all computations in single precision
typedef float real; 
typedef int integer;
typedef int logical;

#ifndef NULL
#define NULL ((char *)0)
#endif

int assert(int a) {
  char *null=NULL;
  if (!a) { 
    if (*null) return 0;// attempt to read from invalid address zero
  }
  return 1;
 }

//typedef real (*E_fp)();
typedef char *E_fp;  // Don't use this anymore... hardwired to funct_(...)

float dabs(float p) { return fabs(p); }
float dmax(float p,float q) { if (p > q) return p;else return q; }
float dmin(float p,float q) { if (p < q) return p;else return q; }

doublereal pow_dd(doublereal *arg1,const __constant doublereal *arg2)
{
  return pow(*arg1,*arg2);
}

/* C source for R1MACH -- remove the * in column 1 */
doublereal r1mach_(const __constant integer *i)
{
	switch(*i){
	  case 1: return FLT_MIN;
	  case 2: return FLT_MAX;
	  case 3: return FLT_EPSILON/FLT_RADIX;
	  case 4: return FLT_EPSILON;
	  case 5: return log10((float)FLT_RADIX);
	  }
        printf("invalid argument: r1mach(%%ld)\n", *i);
        return 0.0f;
//	assert(0); return 0; /* else complaint of missing return value */
}


#define TRUE_ 1
#define FALSE_ 0

#define integrand_yintegral theta_integral


#define LIMIT 7000  // max # of integration intervals

#define qagse_ qagse_yintegral
#define qelg_ qelg_yintegral
#define qk21_ qk21_yintegral
#define qpsrt_ qpsrt_yintegral
#define funct_ integrand_yintegral
#define c__4 c__4_yintegral
#define c__1 c__1_yintegral
#define c__2 c__2_yintegral
#define c_b42 c_b42_yintegral
#define xgk xgk_yintegral
#define wgk wgk_yintegral
#define wg wg_yintegral

%s 

#undef qagse_ 
#undef qelg_ 
#undef qk21_ 
#undef qpsrt_ 
#undef funct_ 
#undef c__4
#undef c__1
#undef c__2
#undef c_b42
#undef xgk
#undef wgk
#undef wg



#define qagse_ qagse_thetaintegral
#define qelg_ qelg_thetaintegral
#define qk21_ qk21_thetaintegral
#define qpsrt_ qpsrt_thetaintegral
#define funct_ integrand_thetaintegral
#define c__4 c__4_thetaintegral
#define c__1 c__1_thetaintegral
#define c__2 c__2_thetaintegral
#define c_b42 c_b42_thetaintegral
#define xgk xgk_thetaintegral
#define wgk wgk_thetaintegral
#define wg wg_thetaintegral

%s 

#undef qagse_ 
#undef qelg_ 
#undef qk21_ 
#undef qpsrt_ 
#undef funct_ 
#undef c__4
#undef c__1
#undef c__2
#undef c_b42
#undef xgk
#undef wgk
#undef wg


#define qagse_ qagse_rintegral
#define qelg_ qelg_rintegral
#define qk21_ qk21_rintegral
#define qpsrt_ qpsrt_rintegral
#define funct_ integrand_Rintegral
#define c__4 c__4_rintegral
#define c__1 c__1_rintegral
#define c__2 c__2_rintegral
#define c_b42 c_b42_rintegral
#define xgk xgk_rintegral
#define wgk wgk_rintegral
#define wg wg_rintegral

%s 

#undef qagse_ 
#undef qelg_ 
#undef qk21_ 
#undef qpsrt_ 
#undef funct_ 
#undef c__4
#undef c__1
#undef c__2
#undef c_b42


float Tcrack(float R,float t, float t1, float t2, float alpha)
{
  float factor = 1.0/(R);
  float term1, term2;

  if (t <= t1) {
    return 0.0;
  }
  
  term1=erfc(R/sqrt(4.0f*alpha*(t-t1)));
  
  if (t <= t2) {
    term2=0.0;
  } else {
    term2=-erfc(R/sqrt(4.0f*alpha*(t-t2)));
  }
  //printf("R=%%f,alpha=%%f,t=%%f\n",R,alpha,t);
  //printf("Tcrack=%%f; term1_param=%%f\n",factor*(term1+term2),R/sqrt(4.0f*alpha*(t-t1)));

  return factor*(term1+term2);
}

// crack center hardwired to the origin
__constant const float crackcenter_x=0.0f;
__constant const float crackcenter_y=0.0f;

doublereal integrand_Rintegral(float *R, float *theta, float *position_x,float *position_y,float *t, float *t1, float *t2, float *alpha,float *junk1)
{
  float retval;

  float totalR;
  //printf("integrand_R=%%f; theta=%%f, position_x=%%f\n",*R,*theta,*position_x);

  totalR = sqrt(pow(crackcenter_x + (*R)*sin(*theta) - (*position_x),2.0f) + pow(crackcenter_y-(*position_y),2.0f) + pow((*R)*cos(*theta),2.0f));
  retval=Tcrack(totalR,*t,*t1,*t2,*alpha)*(*R);

  //printf("R_integrand=%%f\n",retval);
  return retval;
}


doublereal integrand_thetaintegral(float *theta, float *position_x,float *position_y,float *t, float *t1, float *t2, float *alpha,float *R1,float *R2)
{
  float epsabs = 1.0f/(1e5f*(fabs(*R1)+fabs(*R2))/2.0f);
  float epsrel = 1e-6f; 

  int limit = LIMIT; 
  float result=0.0;
  float abserr=0.0;
  int neval=0;
  int ier=0;
  float alist[LIMIT];
  float blist[LIMIT];
  float rlist[LIMIT];
  float elist[LIMIT];
  int iord[LIMIT];
  int last=0;
  float zero=0.0f;
  float v1,v2;


  //printf("R1=%%f; R2=%%f, position_x=%%f\n",*R1,*R2,*position_x);
  qagse_rintegral(NULL,theta,position_x,position_y,t,t1,t2,alpha,&zero,
                         R1,R2,
                         &epsabs,&epsrel,
                         &limit,
                         &result,
                         &abserr,
                         &neval,
                         &ier,
                         alist,blist,rlist,elist,iord,
                         &last);
  v1=Tcrack(sqrt(pow(crackcenter_x + (*R1)*sin(*theta) - (*position_x),2.0f) + pow(crackcenter_y-(*position_y),2.0f) + pow((*R1)*cos(*theta),2.0f)),*t,*t1,*t2,*alpha);
  v2=Tcrack(sqrt(pow(crackcenter_x + (*R2)*sin(*theta) - (*position_x),2.0f) + pow(crackcenter_y-(*position_y),2.0f) + pow((*R2)*cos(*theta),2.0f)),*t,*t1,*t2,*alpha);

  // troubleshoot any errors except for ier==3 where the integral is 
  // essentially zero.
  if (ier != 0 && !(ier==3 && fabs(v1) < 1e-6) && fabs(v2) < 1e-6) {
    printf("rintegral: ier=%%d; R1=%%f R2=%%f pos_x=%%f pos_y=%%f t=%%f, epsabs=%%g Tcrack1=%%g,Tcrack2=%%g\n",ier,*R1,*R2,*position_x,*position_y,*t,epsabs,v1,v2);
  }
  //printf("rintegral=%%g\n",result);
  return result;
}

// Note: y position is first parameter to theta_integral so we can also use it as an integrand directly
doublereal theta_integral(float *pos_y,float *pos_x,float *R1,float *R2,float *t,float *t1,float *t2,float *alpha,float *posside_float)
{
  float epsabs = 1.0f/(1e5f*(fabs(*R1)+fabs(*R2))/2.0f);
  float epsrel = 1e-6f; 
  int limit = LIMIT; 
  float result=0.0;


  float abserr=0.0;
  int neval=0;
  int ier=0;
  float alist[LIMIT];
  float blist[LIMIT];
  float rlist[LIMIT];
  float elist[LIMIT];
  int iord[LIMIT];
  int last=0;
  float theta1,theta2;
  float v1,v2;


  if (*posside_float > 0.5f) {
    theta1=0.0f; // *.99f;  
    theta2=(M_PI_F/2.0f); // *.99f;
  } else {
    theta1=(-M_PI_F/2.0f); // *.99f;  
    theta2=0.0f; // *.99f;

  }

  //printf("integrate: R1=%%f; R2=%%f, position_x=%%f\n",R1val,R2val,pos_x);
  qagse_thetaintegral(NULL,pos_x,pos_y,t,t1,t2,alpha,R1,R2,
                         &theta1,&theta2,
                         &epsabs,&epsrel,
                         &limit,
                         &result,
                         &abserr,
                         &neval,
                         &ier,
                         alist,blist,rlist,elist,iord,
                         &last);

  
  if (ier != 0) {
    v1=integrand_thetaintegral(&theta1,pos_x,pos_y,t, t1,t2,alpha,R1,R2);
    v2=integrand_thetaintegral(&theta1,pos_x,pos_y,t, t1,t2,alpha,R1,R2);
    printf("thetaintegral: ier=%%d; R1=%%f R2=%%f pos_x=%%f pos_y=%%f t=%%f, epsabs=%%g Tcrack1=%%g,Tcrack2=%%g\n",ier,*R1,*R2,*pos_x,*pos_y,*t,epsabs,v1,v2);
  }

  return result;
}




float y_integral(float refpos_y,float infpos_y,float pos_x,float R1,float R2,float t,float t1,float t2,float alpha,int posside)
// refpos_y should be a relevant distance from y=0... suggest 2mm
// infpos_y should be a large distance from y=0... suggest 2m
// we will integrate from 0...refpos_y, then from refpos_y to infpos_y, 
// then double the result to approximate the integral from -infinity to +infinity
{
  float epsabs = 1.0f/(1e5f*(fabs(R1)+fabs(R2))/2.0f);
  float epsrel = 1e-6f; 
  int limit = LIMIT; 
  float result1=0.0,result2=0.0;


  float abserr=0.0;
  int neval=0;
  int ier=0;
  float alist[LIMIT];
  float blist[LIMIT];
  float rlist[LIMIT];
  float elist[LIMIT];
  int iord[LIMIT];
  int last=0;
  float y1,y2,y3;
  float posside_float=(posside != 0) ? 1.0:0.0; 

  y1=0.0;
  y2=refpos_y;
  y3=infpos_y;
  

  qagse_yintegral(NULL,&pos_x,&R1,&R2,&t,&t1,&t2,&alpha,&posside_float,
                         &y1,&y2,
                         &epsabs,&epsrel,
                         &limit,
                         &result1,
                         &abserr,
                         &neval,
                         &ier,
                         alist,blist,rlist,elist,iord,
                         &last);

  if (ier != 0) {
    printf("yintegral: ier=%%d\n",ier);
  }

  qagse_yintegral(NULL,&pos_x,&R1,&R2,&t,&t1,&t2,&alpha,&posside_float,
                         &y2,&y3,
                         &epsabs,&epsrel,
                         &limit,
                         &result2,
                         &abserr,
                         &neval,
                         &ier,
                         alist,blist,rlist,elist,iord,
                         &last);

  if (ier != 0) {
    printf("yintegral: ier=%%d\n",ier);
  }

  return 2.0*(result1+result2);
}



__kernel void integrate_r_theta(__global const float *position_x,
                        __global const float *position_y,
                        __global const float *R1,
                        __global const float *R2,
                        __global const float *t,
                        __global float *T,
                        float t1,float t2, float alpha,
                        int posside)
{
  int gid=get_global_id(0);
  float result;
  float pos_x=position_x[gid];
  float pos_y=position_y[gid];
  float R1val=R1[gid];
  float R2val=R2[gid];
  float tval=t[gid];
  float posside_float=(posside != 0) ? 1.0:0.0; 

  result = theta_integral(&pos_y,&pos_x,&R1val,&R2val,&tval,&t1,&t2,&alpha,&posside_float);

  T[gid]=result;

}



__kernel void integrate_r_theta_y(__global const float *position_x,
                        __global const float *R1,
                        __global const float *R2,
                        __global const float *t,
                        __global float *T,
                        float t1,float t2, float alpha,
                        float refpos_y,float infpos_y,unsigned iteridx,int posside)
{
  int gid=get_global_id(0);
  float result;
  float pos_x=position_x[gid+iteridx];
  float R1val=R1[gid+iteridx];
  float R2val=R2[gid+iteridx];
  float tval=t[gid+iteridx];
  float junk=0.0;

  result = y_integral(refpos_y,infpos_y,pos_x,R1val,R2val,tval,t1,t2,alpha,posside);

  T[gid+iteridx]=result;

}



"""

# find current module so we can use path to load "qagse_fparams.c"
class dummy(object):
    pass
modpath = sys.modules[dummy.__module__].__file__
moddir = os.path.split(modpath)[0]



qagse_fparams=open(os.path.join(moddir,"qagse_fparams.c"),"r").read()

kernelcode=kernelpattern % (qagse_fparams,qagse_fparams,qagse_fparams)

def surface_heating(x_nd,y_nd,t_nd,stripradius1,stripradius2,t1,t2,alpha,k,posside,ctx=None):

    if ctx is None:
        ctx = cl.create_some_context()
        pass
    queue = cl.CommandQueue(ctx)

    (x_bc,y_bc,t_bc,R1_bc,R2_bc)=np.broadcast_arrays(x_nd,y_nd,t_nd,stripradius1,stripradius2)
    
    T_nd=np.zeros(x_bc.shape,dtype='f')


    x_unwrap=np.ascontiguousarray(x_bc.astype(np.float32).ravel())
    y_unwrap=np.ascontiguousarray(y_bc.astype(np.float32).ravel())
    t_unwrap=np.ascontiguousarray(t_bc.astype(np.float32).ravel())
    R1_unwrap=np.ascontiguousarray(R1_bc.astype(np.float32).ravel())
    R2_unwrap=np.ascontiguousarray(R2_bc.astype(np.float32).ravel())

    mf = cl.mem_flags
    x_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=x_unwrap)
    y_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=y_unwrap)
    t_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=t_unwrap)
    R1_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=R1_unwrap)
    R2_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=R2_unwrap)
    T_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=T_nd.nbytes)
    
    prg=cl.Program(ctx,kernelcode)
    prg.build()

    integrate_r_theta_kernel=prg.integrate_r_theta

    integrate_r_theta_kernel.set_scalar_arg_dtypes([
        None,None,None,None,None,None,
        np.float32,np.float32,np.float32,np.int32
    ])
    
    # Run the compute kernel
    res=integrate_r_theta_kernel(queue,x_unwrap.shape,None,
                                 x_buf,y_buf,
                                 R1_buf,R2_buf,
                                 t_buf,
                                 T_buf,
                                 t1,t2,alpha,posside)
    cl.enqueue_copy(queue,T_nd,T_buf,wait_for=(res,),is_blocking=True)

    return T_nd/(2.0*np.pi*k) # multiply by leading factor of 1.0/(2.0*pi*k).. see comment at top of file





def surface_heating_y_integral(refpos_y,infpos_y,x_nd,t_nd,stripradius1,stripradius2,t1,t2,alpha,k,posside,ctx=None,maxiter=2000):

    if ctx is None:
        ctx = cl.create_some_context()
        pass
    queue = cl.CommandQueue(ctx)

    (x_bc,t_bc,R1_bc,R2_bc)=np.broadcast_arrays(x_nd,t_nd,stripradius1,stripradius2)
    
    yint_nd=np.zeros(x_bc.shape,dtype='f')


    x_unwrap=np.ascontiguousarray(x_bc.astype(np.float32).ravel())
    t_unwrap=np.ascontiguousarray(t_bc.astype(np.float32).ravel())
    R1_unwrap=np.ascontiguousarray(R1_bc.astype(np.float32).ravel())
    R2_unwrap=np.ascontiguousarray(R2_bc.astype(np.float32).ravel())

    mf = cl.mem_flags
    x_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=x_unwrap)
    t_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=t_unwrap)
    R1_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=R1_unwrap)
    R2_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=R2_unwrap)
    yint_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=yint_nd.nbytes)
    
    prg=cl.Program(ctx,kernelcode)
    prg.build()

    integrate_r_theta_y_kernel=prg.integrate_r_theta_y

    integrate_r_theta_y_kernel.set_scalar_arg_dtypes([
        None,None,None,None,None,
        np.float32,np.float32,np.float32,np.float32,np.float32,np.uint32,np.int32
    ])
    
    # Run the compute kernel, splitting into maxiter chunks
    
    for itercnt in range((x_unwrap.shape[0]+maxiter-1)//maxiter):
        iteridx = maxiter * itercnt;
        niter = maxiter
        if niter+iteridx > x_unwrap.shape[0]:
            niter = x_unwrap.shape[0]-iteridx
            pass
        
        res=integrate_r_theta_y_kernel(queue,(niter,),None,
                                       x_buf,
                                       R1_buf,R2_buf,
                                       t_buf,
                                       yint_buf,
                                       t1,t2,alpha,
                                       refpos_y,infpos_y,iteridx,posside)
        pass
    
    cl.enqueue_copy(queue,yint_nd,yint_buf,wait_for=(res,),is_blocking=True)
        
    return yint_nd/(2.0*np.pi*k) # multiply by leading factor of 1.0/(2.0*pi*k).. see comment at top of file


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

    pl.figure()
    pl.imshow(T_nd[:,:,2])


    x_2=np.arange(-10e-3,10e-3,.1e-3)
    t_2=np.arange(0.0,2.2,.0109)

    refpos_y=2.0e-3
    infpos_y=2.0
    
    (x_2_nd,t_2_nd)=np.meshgrid(x_2,t_2,indexing="ij")
    yint_nd=surface_heating_y_integral(refpos_y,infpos_y,x_2_nd,t_2_nd,stripradius1,stripradius2,t1,t2,alpha,k,posside)  
    pl.figure()
    pl.imshow(yint_nd)




    
    pl.show()
    pass

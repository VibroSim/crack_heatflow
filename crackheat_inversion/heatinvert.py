
# Heatinvert.py: Given heating as a function of position x along
# crack, evaluate heating

def heatinvert(x,xcenter,t,t1,t2,k,rhoc,strip_r_bnds,refpos_y,infpos_y,Tdata,tikparam):
    """ Invert observed heating on the assumption that the 
heating originates from quarter-circular bands centered at
xcenter. Tdata is should be a function of x (axis 0, direction along crack) 
and t (axis 1) and integrated along y (direction across crack)

Tikparam may need to be tuned if the time base changes, but should be 
invariant under spatial changes. 
"""

    ctx=cl.create_some_context()

    alpha=k/float(rhoc)

    nx=x.shape[0]
    nt=t.shape[0]

    assert((nx,nt)==Tdata.shape)
    n_bnds = strip_r_bnds.shape[0]-1
    dr = r_bnds[1]-r_bnds[0]

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


    # Make Tikhonov parameter scale invariant
    # by pre-scaling A by dx... then we will divide U by dx
    # likewise we will pre-scale A by 1/dr and multiply V by dr
    # ... see greensinversion paper and greensinversion's sourcevecs.py
    # for complete rationale

    inversion_mtx_scaled = inversion_mtx * dx/dr

    (u,s,v)=np.linalg.svd(inversion_mtx_scaled,full_matrices=False)   # inversion/t_amount
    
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


    # Do tikhonov regularized inversion
    d=s/(s**2.0+tikparam**2.0)
    
    bestfit = np.dot(v_scaled.T,np.dot(u_scaled.T,Tdata.reshape(nx*nt))*d)

    recon=np.dot(inversion_mtx,bestfit).reshape(nx,nt)

    return (bestfit,recon,s)
    

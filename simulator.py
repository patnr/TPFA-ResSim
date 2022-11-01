"""From [ref](http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf)"""
import numpy as np
from scipy import sparse


def Pres(Grid, S, Fluid, q):
    # Compute K*lambda(S)
    Mw, Mo = RelPerm(S, Fluid)
    Mt = Mw+Mo
    Mt = Mt.reshape((Grid['Nx'], Grid['Ny'], Grid['Nz']), order="F")
    KM = Mt*Grid['K']
    # Compute pressure and extract fluxes
    [P, V] = TPFA(Grid, KM, q)
    return P, V


def RelPerm(s, Fluid, nargout_is_4=False):
    S = (s-Fluid['swc'])/(1-Fluid['swc']-Fluid['sor']) # Rescale saturations
    Mw = S**2/Fluid['vw']                              # Water mobility
    Mo = (1-S)**2/Fluid['vo']                          # Oil mobility
    if nargout_is_4:
        dMw = 2*S/Fluid['vw']/(1-Fluid['swc']-Fluid['sor'])
        dMo = -2*(1-S)/Fluid['vo']/(1-Fluid['swc']-Fluid['sor'])
        return Mw, Mo, dMw, dMo
    else:
        return Mw, Mo


def TPFA(Grid,K,q):
    # Compute transmissibilities by harmonic averaging.
    Nx=Grid['Nx']
    Ny=Grid['Ny']
    Nz=Grid['Nz']
    N=Nx*Ny*Nz

    hx=Grid['hx']
    hy=Grid['hy']
    hz=Grid['hz']

    L = K**(-1)

    tx = 2*hy*hz/hx
    TX = np.zeros((Nx+1,Ny,Nz))

    ty = 2*hx*hz/hy
    TY = np.zeros((Nx,Ny+1,Nz))

    tz = 2*hx*hy/hz
    TZ = np.zeros((Nx,Ny,Nz+1))


    TX[1:-1,:,:] = tx/(L[0,:-1,:,:] + L[0,1:,:,:])
    TY[:,1:-1,:] = ty/(L[1,:,:-1,:] + L[1,:,1:,:])
    TZ[:,:,1:-1] = tz/(L[2,:,:,:-1] + L[2,:,:,1:])


    # Assemble TPFA discretization matrix.
    x1 = TX[:-1,:,:].ravel(order="F")
    x2 = TX[1:,:,:].ravel(order="F")

    y1 = TY[:,:-1,:].ravel(order="F")
    y2 = TY[:,1:,:].ravel(order="F")

    z1 = TZ[:,:,:-1].ravel(order="F")
    z2 = TZ[:,:,1:].ravel(order="F")


    DiagVecs = [   -z2, -y2, -x2, x1+x2+y1+y2+z1+z2, -x1, -y1,  -z1]
    DiagIndx = [-Nx*Ny, -Nx,  -1,         0        ,  1 ,  Nx, Nx*Ny]


    A = sparse.spdiags(DiagVecs, DiagIndx, N, N)
    A = A.toarray()
    A[0, 0] = A[0, 0]+np.sum(Grid['K'][:, 0, 0, 0])

    # Solve linear system and extract interface fluxes.
    u = np.linalg.solve(A, q)
    P = u.reshape((Nx, Ny, Nz), order="F")

    V = {}
    V['x'] = np.zeros((Nx+1, Ny, Nz))
    V['y'] = np.zeros((Nx, Ny+1, Nz))
    V['z'] = np.zeros((Nx, Ny, Nz+1))

    V['x'][1:-1, :, :] = (P[:-1, :, :] - P[1:, :, :]) * TX[1:-1, :, :]
    V['y'][:, 1:-1, :] = (P[:, :-1, :] - P[:, 1:, :]) * TY[:, 1:-1, :]
    V['z'][:, :, 1:-1] = (P[:, :, :-1] - P[:, :, 1:]) * TZ[:, :, 1:-1]
    return P, V


def GenA(Grid, V, q):
    Nx = Grid['Nx']
    Ny = Grid['Ny']
    Nz = Grid['Nz']
    N = Nx*Ny*Nz
    fp = q.clip(max=0).ravel()  # production
    # Flow fluxes, separated into direction (x-y) and sign
    XN=V['x'].clip(max=0); x1=XN[:-1, :, :].ravel(order="F")
    YN=V['y'].clip(max=0); y1=YN[:,:-1,:]  .ravel(order="F")
    ZN=V['z'].clip(max=0); z1=ZN[:,:,:-1]  .ravel(order="F")
    XP=V['x'].clip(min=0); x2=XP[1:,:,:]   .ravel(order="F")
    YP=V['y'].clip(min=0); y2=YP[:,1:,:]   .ravel(order="F")
    ZP=V['z'].clip(min=0); z2=ZP[:,:,1:]   .ravel(order="F")
    DiagVecs=[    z2,  y2, x2, fp+x1-x2+y1-y2+z1-z2, -x1, -y1, -z1]    # diagonal vectors
    DiagIndx=[-Nx*Ny, -Nx, -1,           0         ,  1 ,  Nx, Nx*Ny]  # diagonal index
    # matrix with upwind FV stencil
    A=sparse.spdiags(DiagVecs,DiagIndx,N,N)
    return A


def Upstream(Grid,S,Fluid,V,q,T):
    Nx = Grid['Nx']
    Ny = Grid['Ny']
    Nz = Grid['Nz']                                        # number of grid points
    N = Nx*Ny*Nz                                           # number of unknowns
    # pore volume=cell volume*porosity
    pv = Grid['V']*Grid['por'].ravel(order="F")

    # Well inflow
    fi = q.clip(min=0)

    # In-/Out-flux x-/y- faces
    XP=V['x'].clip(min=0)
    XN=V['x'].clip(max=0)
    YP=V['y'].clip(min=0)
    YN=V['y'].clip(max=0)
    ZP=V['z'].clip(min=0)
    ZN=V['z'].clip(max=0)

    Vi = XP[:-1,:,:]+YP[:,:-1,:]+ZP[:,:,:-1]- \
         XN[ 1:,:,:]-YN[:, 1:,:]-ZN[:,:, 1:]  # each gridblock

    # Compute dt
    pm = min(pv/(Vi.ravel(order="F")+fi.ravel()))          # estimate of influx
    cfl = ((1-Fluid['swc']-Fluid['sor'])/3)*pm             # CFL restriction
    Nts = int(np.ceil(T/cfl))                              # number of local time steps
    dtx = (T/Nts)/pv                                       # local time steps
    A = GenA(Grid, V, q)                                   # system matrix
    A = sparse.spdiags(dtx, 0, N, N)@A                     # A * dt/|Omega i|
    fi = q.clip(min=0).ravel()*dtx                         # injection
    for t in range(1, Nts+1):
        mw, mo = RelPerm(S, Fluid)                         # compute mobilities
        fw = mw/(mw+mo)                                    # compute fractional flow
        S = S+(A@fw+fi[:, None])                           # update saturation
    return S


if __name__ == "__main__":
    # Settings as in listing 9
    Dx = 1
    Dy = 1
    Dz = 1
    Grid = dict(
        Nx = 64,
        Ny = 64,
        Nz = 1 ,
    )
    Grid['hx'] = Dx/Grid['Nx'] # Dimension in x-direction
    Grid['hy'] = Dy/Grid['Ny'] # Dimension in y-direction
    Grid['hz'] = Dz/Grid['Nz'] # Dimension in z-direction
    N=Grid['Nx']*Grid['Ny']    # Total number of grid blocks

    Grid['V'] = Grid['hx']*Grid['hy']*Grid['hz']                 # Cell volumes
    Grid['K'] = np.ones((3, Grid['Nx'], Grid['Ny'], Grid['Nz'])) # Unit permeability
    Grid['por'] = np.ones((Grid['Nx'], Grid['Ny'], Grid['Nz']))  # Unit porosity

    # Source terms: production/injection
    Q     = np.zeros(Grid.N)
    Q[0]  = +1
    Q[-1] = -1
    Q     = Q[:, None]

    Fluid = {}
    Fluid['vw']=1.0; Fluid['vo']=1.0; # Viscosities
    Fluid['swc']=0.0; Fluid['sor']=0.0; # Irreducible saturations

    # nt=28 used in paper
    nt = 2
    S=np.zeros(N)[:,None]; # Initial saturation

    dt = 0.7/nt
    for t in range(1,nt+1):
        [P,V]=Pres(Grid,S,Fluid,Q); # pressure solver
        S=Upstream(Grid,S,Fluid,V,Q,dt); # saturation solver

    # I have cross-checked the output of this code with that of the Matlab code,
    # and ensured that they produce the same values. Example locations/values:
    assert np.isclose(S[20]  , 0.94726344036)
    assert np.isclose(S[1300], 0.90674468214)
    assert np.isclose(S[2900], 0.794624098299)

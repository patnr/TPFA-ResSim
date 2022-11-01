% From article, listing 8
function S=Upstream(Grid,S,Fluid,V,q,T)

    Nx=Grid.Nx; Ny=Grid.Ny; Nz=Grid.Nz;               % number of grid points
    N=Nx*Ny*Nz;                                       % number of unknowns
    pv = Grid.V(:).*Grid.por (:);                     % pore volume=cell volume*porosity

    fi =max(q,0);                                     % inflow from wells
    XP=max(V.x,0); XN=min(V.x,0);                     % influx and outflux, x-faces
    YP=max(V.y,0); YN=min(V.y,0);                     % influx and outflux, y-faces
    ZP=max(V.z,0); ZN=min(V.z,0);                     % influx and outflux, z-faces

    Vi = XP(1:Nx,:,:)+YP(:,1:Ny,:)+ZP(:,:,1:Nz)-...   % total flux into
        XN(2:Nx+1,:,:)-YN(:,2:Ny+1,:)-ZN(:,:,2:Nz+1); % each gridblock
    pm = min(pv./(Vi(:)+fi));                         % estimate of influx
    cfl = ((1-Fluid.swc-Fluid.sor)/3)*pm;             % CFL restriction
    Nts = ceil(T/cfl );                               % number of local time steps
    dtx = (T/Nts)./pv;                                % local time steps

    A=GenA(Grid,V,q);                                 % system matrix
    A=spdiags(dtx,0,N,N)*A;                           % A * dt/|Omega i|
    fi =max(q,0).*dtx;                                % injection

    for t=1:Nts
        [mw,mo]=RelPerm(S,Fluid);                     % compute mobilities
        fw = mw./(mw+mo);                             % compute fractional flow
        S = S+(A*fw+fi);                              % update saturation
    end

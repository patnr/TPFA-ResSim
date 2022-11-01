% From article, listing 7
function A=GenA(Grid,V,q)

    Nx=Grid.Nx; Ny=Grid.Ny; Nz=Grid.Nz; N=Nx*Ny*Nz;
    N=Nx*Ny*Nz;  % number of unknowns
    fp=min(q,0); % production

    XN=min(V.x,0); x1=reshape(XN(1:Nx,:,:),N,1);   % separate flux into
    YN=min(V.y,0); y1=reshape(YN(:,1:Ny,:),N,1);   % - flow in positive coordinate
    ZN=min(V.z,0); z1=reshape(ZN(:,:,1:Nz),N,1);   %   direction (XP,YP,ZP)
    XP=max(V.x,0); x2=reshape(XP(2:Nx+1,:,:),N,1); % - flow in negative coordinate
    YP=max(V.y,0); y2=reshape(YP(:,2:Ny+1,:),N,1); %   direction (XN,YN,ZN)
    ZP=max(V.z,0); z2=reshape(ZP(:,:,2:Nz+1),N,1); %

    DiagVecs=[z2,y2,x2,fp+x1-x2+y1-y2+z1-z2,-x1,-y1,-z1]; % diagonal vectors
    DiagIndx=[-Nx*Ny,-Nx,-1,0,1,Nx,Nx*Ny];                % diagonal index
    A=spdiags(DiagVecs,DiagIndx,N,N);                     % matrix with upwind FV stencil

% From article, listing 9
% Modified to also propose "implicit" saturation step,
% which is faster when the grid is high-resolution.

implicit = 1;

Grid.Nx=64; Dx=1; Grid.hx = Dx/Grid.Nx;  % Dimension in x-direction
Grid.Ny=64; Dy=1; Grid.hy = Dy/Grid.Ny;  % Dimension in y-direction
Grid.Nz=1;  Dz=1; Grid.hz = Dz/Grid.Nz;  % Dimension in z-direction
N=Grid.Nx*Grid.Ny;                       % Total number of grid blocks
Grid.V=Grid.hx*Grid.hy*Grid.hz;          % Cell volumes
Grid.K=ones(3,Grid.Nx,Grid.Ny,Grid.Nz);  % Unit permeability
Grid.por =ones(Grid.Nx,Grid.Ny,Grid.Nz); % Unit porosity
Q=zeros(N,1); Q([1 N])=[1 -1];           % Production/injection

% Viscosities (line1), Irreducible saturations (line2)
Fluid.vw=1; Fluid.vo=1;
Fluid.swc=0; Fluid.sor=0;
% As in listing 11
% Fluid.vw=3e-4; Fluid.vo=3e-3;
% Fluid.swc=0.2; Fluid.sor=0.2;

tic
S=Fluid.swc*ones(N,1); % Initial saturation
nt = 28; dt = 0.7/nt;  % Time steps
for t=1:nt
    % Pressure
    [P,V]=Pres(Grid,S,Fluid,Q);
    % Saturation
    if implicit
      S=NewtRaph(Grid,S,Fluid,V,Q,dt);
    else
      S=Upstream(Grid,S,Fluid,V,Q,dt);
    endif

    % plot filled contours at the midpoints of the grid cells
    figure(1)
    contourf(linspace(Grid.hx/2,Dx-Grid.hx/2,Grid.Nx),...
      linspace(Grid.hy/2,Dy-Grid.hy/2,Grid.Ny),...
      reshape(S,Grid.Nx,Grid.Ny),11,'k');
    axis square; caxis ([0 1]);
    title(["t = " num2str(t*dt)])

    % Force update of plot
    % pause(.5)
    drawnow;
end
toc

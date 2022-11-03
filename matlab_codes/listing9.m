% From article, listing 9
Grid.Nx=64; Dx=1; Grid.hx = Dx/Grid.Nx;  % Dimension in x-direction
Grid.Ny=64; Dy=1; Grid.hy = Dy/Grid.Ny;  % Dimension in y-direction
Grid.Nz=1; Dz=1; Grid.hz = Dz/Grid.Nz;   % Dimension in z-direction
N=Grid.Nx*Grid.Ny;                       % Total number of grid blocks
Grid.V=Grid.hx*Grid.hy*Grid.hz;          % Cell volumes
Grid.K=ones(3,Grid.Nx,Grid.Ny,Grid.Nz);  % Unit permeability
Grid.por =ones(Grid.Nx,Grid.Ny,Grid.Nz); % Unit porosity
Q=zeros(N,1); Q([1 N])=[1 -1];           % Production/injection

Fluid.vw=1.0; Fluid.vo=1.0;              % Viscosities
Fluid.swc=0.0; Fluid.sor=0.0;            % Irreducible saturations

S=zeros(N,1);                            % Initial saturation
nt = 28; dt = 0.7/nt;                    % Time steps
for t=1:nt
    [P,V]=Pres(Grid,S,Fluid,Q);          % pressure solver
    S=Upstream(Grid,S,Fluid,V,Q,dt);     % saturation solver

    % plot filled contours at the midpoints of the grid cells
    contourf(linspace(Grid.hx/2,Dx-Grid.hx/2,Grid.Nx),...
    linspace(Grid.hy/2,Dy-Grid.hy/2,Grid.Ny),...
    reshape(S,Grid.Nx,Grid.Ny),11,'k');
    axis square ; caxis ([0 1]);        % equal axes and color
    %pause(.5)
    drawnow;                            % force update of plot
end

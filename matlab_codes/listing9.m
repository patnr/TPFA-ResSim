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

% For reproduction/test purposes:
% >> format long
% >> S(1:100:end)
% ans =
%
%    9.996533607460576e-01
%    8.660858151181138e-01
%    9.861756811920124e-01
%    8.147484733167993e-01
%    9.575534390194773e-01
%    7.456430889847783e-01
%    9.196770579509945e-01
%    2.202292655742370e-03
%    8.767170514215988e-01
%    9.675365926352755e-01
%    8.299769901829444e-01
%    9.412283901123030e-01
%    7.758428865277260e-01
%    9.068072128562447e-01
%    6.572565198694793e-01
%    8.687214455426303e-01
%    9.245222553603517e-01
%    8.292032843264070e-01
%    9.026812050974996e-01
%    7.873032429511491e-01
%    8.738854571188687e-01
%    7.315182756915809e-01
%    8.422825553657579e-01
%    6.136526207843503e-04
%    8.106786714921117e-01
%    8.469404137187946e-01
%    7.800336973393258e-01
%    8.218406165872928e-01
%    7.477238908589862e-01
%    7.941048496654481e-01
%    6.746706375814319e-01
%    7.658865427603075e-01
%    7.682937266433443e-01
%    7.378812306524954e-01
%    7.325909173584360e-01
%    7.084295240195730e-01
%    6.606542343162950e-01
%    6.687276573733374e-01
%    6.812542532586809e-02
%    5.876144676769273e-01
%    9.108992794669927e-20

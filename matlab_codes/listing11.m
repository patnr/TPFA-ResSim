% From article, listing 11,
% but mixed with listing 9 (quarter 5-spot)
% Unfortunately, possibly due to the stiffness of this case,
% the first step takes forever to run (does it even converge?).

Grid.Nx=64; Dx=1; Grid.hx = Dx/Grid.Nx;  % Dimension in x-direction
Grid.Ny=64; Dy=1; Grid.hy = Dy/Grid.Ny;  % Dimension in y-direction
Grid.Nz=1; Dz=1; Grid.hz = Dz/Grid.Nz;   % Dimension in z-direction
N=Grid.Nx*Grid.Ny;                       % Total number of grid blocks
Grid.V=Grid.hx*Grid.hy*Grid.hz;          % Cell volumes
Grid.K=ones(3,Grid.Nx,Grid.Ny,Grid.Nz);  % Unit permeability
Grid.por =ones(Grid.Nx,Grid.Ny,Grid.Nz); % Unit porosity
Q=zeros(N,1); Q([1 N])=[1 -1];           % Production/injection

Fluid.vw=3e-4; Fluid.vo=3e-3;                          % Viscosities
Fluid.swc=0.2; Fluid.sor=0.2;                          % Irreducible saturations

S=Fluid.swc*ones(N,1);                                 % Initial saturation
Pc=[0; 1]; Tt=0;                                       % For production curves

St = 5;                                                % Maximum saturation time step
Pt = 100;                                              % Pressure time step
ND = 2000;                                             % Number of days in simulation

for tp=1:ND/Pt;
    [P,V]=Pres(Grid,S,Fluid,Q);                        % Pressure solver
    for ts=1:Pt/St;
        S=NewtRaph(Grid,S,Fluid,V,Q,St);               % Implicit saturation solver
        subplot('position' ,[0.05 .1 .4 .8]);          % Make left subplot
        pcolor(reshape(S,Grid.Nx,Grid.Ny,Grid.Nz)');   % Plot saturation
        shading flat; caxis([Fluid.swc 1-Fluid.sor ]); %

        % [Mw,Mo]=RelPerm(S(N),Fluid); Mt=Mw+Mo;         % Mobilities in well-block
        % Tt=[Tt,(tp-1)*Pt+ts*St];                       % Compute simulation time
        % Pc=[Pc,[Mw/Mt; Mo/Mt]];                        % Append production data
        % subplot('position' ,[0.55 .1 .4 .8]);          % Make right subplot
        % plot(Tt,Pc (1,:),Tt,Pc (2,:));                 % Plot production data
        % axis([0,ND,-0.05,1.05]);                       % Set correct axis
        % legend('Water cut','Oil cut' );                % Set legend
        drawnow;                                       % Force update of plot
    end
end

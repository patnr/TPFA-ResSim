% From article, listing 5
function [P,V]=Pres(Grid,S,Fluid,q)

    % Compute K*lambda(S)
    [Mw,Mo]=RelPerm(S,Fluid);
    Mt=Mw+Mo;
    KM = reshape([Mt,Mt,Mt]',3,Grid.Nx,Grid.Ny,Grid.Nz).*Grid.K;

    % Compute pressure and extract fluxes
    [P,V]=TPFA(Grid,KM,q);

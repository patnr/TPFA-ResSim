% From article, listing 10
function S=NewtRaph(Grid,S,Fluid,V,q,T);

N = Grid.Nx*Grid.Ny*Grid.Nz;                         % number of unknowns
A = GenA(Grid,V,q);                                  % system matrix
conv=0; IT=0; S00=S;

while conv==0;
    dt = T/2^IT;                                     % timestep
    dtx = dt./(Grid.V(:)*Grid.por (:));              % timestep / pore volume
    fi = max(q,0).*dtx;                              % injection
    B=spdiags(dtx,0,N,N)*A;

    I=0;
    while I<2^IT;                                    % loop over sub−timesteps
    S0=S; dsn=1; it=0; I=I+1;

        while dsn>1e−3 & it<10;                      % Newton−Raphson iteration
        [Mw,Mo,dMw,dMo]=RelPerm(S,Fluid);            % mobilities and derivatives
        df=dMw./(Mw + Mo)−Mw./(Mw+Mo).^2.*(dMw+dMo); % df w/ds
        dG=speye(N)−B*spdiags(df,0,N,N);             % G’(S)

        fw = Mw./(Mw+Mo);                            % fractional flow
        G = S−S0−(B*fw+fi);                          % G(s)
        ds = −dG\G;                                  % increment ds
        S = S+ds;                                    % update S
        dsn = norm(ds);                              % norm of increment
        it = it+1;                                   % number of N−R iterations
        end

    if dsn>1e−3; I=2^IT; S=S00; end                  % check for convergence
    end

    if dsn<1e−3; conv=1;                             % check for convergence
    else IT=IT+1; end                                % if not converged, decrease
end                                                  % timestep by factor 2

MiniRes is a
2D, two-phase, black-oil, immiscible
reservoir simulator
using TPFA (two-point flux approximation).

![The Egg model: permeability, pressure, oil saturation, and the adjoint sensitivity of a producer's water cut](collage.png)

See `examples` for more demonstrations.

## Governing equations

[1]: https://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

The simulator solves eqn. (1) and (2)
(corresponding to (42) and (43) of the [reference paper][1]) :

$$\begin{align}
    - \nabla \cdot \mathbf{K} \lambda(s) \, \nabla p &= q \,, \tag{1} \cr
    \phi \frac{\partial s}{\partial t} + \nabla \cdot (f(s)\, \mathbf{v}) &= \frac{q_w}{\rho_w} \,. \tag{2}
\end{align}$$

The quantities involved are all 2D-spatial fields, namely

- $\phi \in [0, 1]$ is the porosity
- $s \in [0, 1]$ is the water saturation
- $p$ is the pressure
- $v$ is the (volumetric) flow velocity ($\mathbf{v} = \mathbf{v}_o + \mathbf{v}_w$).
- $q$ is the sources/sinks
- $\rho$ is the density
- $\mathbf{K}$ is the (absolute) permeability tensor: the rock's conductivity to
  flow, here diagonal, $\mathrm{diag}(K_x, K_y)$ per cell.
- $\lambda(s)$ is the total mobility (sum of mobilities).
  Each (relative) mobility is the phase relative permeability
  divided by the phase viscosity, $\lambda_{\text{phase}} = k_{\text{phase}}/\mu_{\text{phase}}$.
  - The relative permeabilities $k_{\text{phase}}(s) \in [0, 1]$ are a *constitutive
    relation*, not data: here, Corey (power-law) curves of the saturation rescaled by
    its residual values, with adjustable exponents and end-points, quadratic by
    default (`ResSim.fluid`, a `minires.fluids.Fluid`). Need not sum to 1.
  - $\mu_{\text{phase}}$ is the phase viscosity, here constant.
- $f(s) = \lambda_w(s) / \lambda(s) \in [0, 1]$ is the water fractional flow,
  giving $\mathbf{v}_w = f(s) \, \mathbf{v}$.

The right hand side of (2) is further simplified (relabelled) as $q$,
i.e. dropping the $w$ (for "water") subscripts.

### Derivation

#### Single phase

**Conservation of mass** in a porous ($\phi$) medium is expressed by

$$\frac{∂(\rho \phi)}{∂t} + ∇ \cdot (\rho \mathbf{v}) = q \,. \tag{3}$$

This equation is also called continuity eqn., advection eqn., transport eqn.,
or even 1st-order wave eqn. (if constant $v$).
It says that divergence (or convergence) must be balanced
by change in density or porosity, or sinks or sources.
If we assume constant porosity, $\phi$, and incompressibility (constant $\rho$),
then the time derivative vanishes, yielding the steady-state equation
$$\nabla \cdot \mathbf{\mathbf{v}} = \frac{q}{\rho} \,. \tag{4}$$

We now have 1 equation and 2 unknowns (in 2D).
Closing the system,
**Darcy's law** provides 2 additional equations and 1 additional unknown, pressure $p$:
$$\mathbf{v} = − \frac{\mathbf{K}}{\mu} \nabla u \,, \tag{5}$$
where
$u = p - \rho g z \,.$
Analogously to Fourier's heat diffusion and Ohm's conduction law,
Darcy's law (5) was initially derived empirically,
but can be shown to be a special case of Navier-Stokes' momentum equation.
It says that $\mathbf{v}$ is the gradient of the *velocity potential*, $u$,
linearly transformed by the permeability tensor (matrix).
Inserting the formula (5) into eqn. (4) yields
$$− \nabla \cdot \frac{\mathbf{K}}{\mu} \nabla u = \frac{q}{\rho} \,. \tag{6}$$
which can be solved for $u$.
In reservoir engineering, **no-flow** boundary conditions are most often used,
and $u$ is only determined up to a constant (as behoves a *potential*).
Finally, $u$ can be inserted in Darcy's law (5) to yield the (steady-state) velocity.

#### Two phases

- Incompressibility again yields eqn. (4) for the *total* (volumetric) velocity.
- Darcy's law (5) is assumed for each (both) individual phase,
  with $\mathbf{K}$ replaced by $\mathbf{K} \lambda_{\text{phase}}(s)$.
- Neglecting $\nabla z$ (gravity, i.e. hydrostatic pressure),
  the flow potential, $u$, reduces to the pressure field, $p$.
- Summing Darcy's law over the two phases yields
  $$\mathbf{v} = − \mathbf{K} \lambda (s) \nabla p \,. \tag{7}$$
- Repeating the steps right above eqn. (6), one arrives at eqn. (1).
- Meanwhile, *immiscibility* means that conservation of mass (3) must hold for each phase separately,
  i.e. the density $\rho$ gets replaced
  by $s_{\text{phase}} \, \rho_{\text{phase}}$,
  and $\mathbf{v}$ by $\mathbf{v}_{\text{phase}} = f_{\text{phase}}(s)\, \mathbf{v}$,
  immediately yielding eqn. (2).

### How to solve

Equations (1) and (2) are nonlinearly coupled:
$s$ and $p$ (yielding $v$ via eqn. (7)) appear in both equations.
Trying to solve both equations simultaneously is a nonlinear root-finding problem,
requiring Newton iterations and matrix inversions.
In this context, it is tempting to use *implicit* time discretization (like ECLIPSE 100)
where $s_{t+1}$ is expressed as a (nonlinear) function of itself,
since this would also requires iterations

Here, instead, we apply sequential operator splitting,
meaning that the two equations are solved independently,
inserting the previous solution of (1) into (2), and vice-versa.
Since it yields smaller systems (which can potentially be discretized explicitly)
this is faster, but less accurate.
When using an explicit (upwind) scheme for the nearly-hyperbolic saturation/transport equation,
the strategy is called IMPES (implicit pressure, explicit saturation).
The simulator also contains implicit saturation scheme,
but it rarely outperforms the explicit one, ref `ResSim.saturation_step_implicit`.

The spatial discretization is carried out by finite volumes (FV),
which is similar to finite differences (FD),
but arguably easier to formulate for non-structured (irregular) grids (not our case).
For the pressure equation, using only two points two approximate the transmissibility
and fluxes at the interfaces is called it is called two-point flux approximation (TPFA);
simple, but used widely (nearly default) in oil industry, due to its robustness and efficiency.
Consider the equation
$$- \nabla \cdot \lambda \nabla u = q \,, \tag{8}$$
where replacing $\lambda \leftarrow \mathbf{K} \lambda(s)$ reproduces eqn. (1),
or $\lambda \leftarrow \mathbf{K}/\mu$ and $q \leftarrow q/\rho$ reproduces eqn. (6).
FV methods apply the divergence theorem to eqn. (8) to replace point derivatives
by integral quantities: interface fluxes and volumetric sources/sinks:
$$- \int_{\partial \Omega_i} d x^2 \, \lambda \, (\nabla u) \cdot \mathbf{n}
= \int_{\Omega_i} d x^3 \, q \,, \tag{9}$$
where $\Omega_i$ is the domain of cell index $i$,
and $\partial \Omega_i$ is its boundary,
with normal vector $\mathbf{n}$.

Now, in TPFA we approximate $(\nabla u) \cdot \mathbf{n}$ by a finite difference
$$\delta u_{ij} := 2 \frac{u_j - u_i}{\Delta x_i + \Delta x_j}$$
where $u_i, u_i$ are the values of the potential, $u$, at *centre* of cells $i$
and $j$, which are located either side of the interface $\gamma_{ij}$,
which is part of $\partial \Omega_i$.
PS: by contrasts, mixed finite-element methods (FEM)
do not approximate fluxes over cell edges but considers them unknown.
Next, $\lambda$ is approximated by a harmonic average, $\lambda_{ij}$,
including weights that account for the distances from the interface to the cell centres.
Thus eqn. (9) becomes
$$- \sum_j |\gamma_{ij}| \lambda_{ij} \delta u_{ij}
= \int_{\Omega_i} d x^3 \, q \,, \tag{10}$$
where the sum is over the indexes $j$ of the interfaces around cell $i$.
The left-hand side can be succinctly expressed as $- \sum_j t_{ij} (u_i - u_j)$,
where $t_{ij}$ (see above their equation 17) is symmetric.
Thus the whole linear system (for all $i$) is symmetric.
Moreover, summing over $i$ yields $\sum_{ij} t_{ij} u_i - \sum_{ij} u_j = 0$,
meaning that the vector of ones is a null vector for the system
(as appropriate for a differential operator),
and that $u$ is determined only up to an arbitrary constant
(as appropriate for a potential).
The constant is fixed, and the system is rendered invertible,
by adding to the first element of the diagonal.

The system is thus symmetric positive definite, and is solved by conjugate
gradients, preconditioned by a sparse LU factorization that is *cached* across
the time steps (`ResSim.cached_precond`).

.. note:: The pressure system need not be factorized afresh each step.

    Its matrix changes only through the mobility $λ(s)$,
    so the factorization of an earlier step is an
    excellent **preconditioner** for the current one, at the cost of
    a back-substitution, and a refactorization only once the iteration stalls.
    `tests/test_precond.py` benchmarks.

### Units

The units are by default SI (m, s, Pa).
But you can switch to metric (m, day, bar, mD)
or field-like (ft, day, psi) by changing `ResSim.cdarcy`.

## Compressibility

The above is the default incompressible model, which is what the reference paper treats.
Below we derive the so-called *slightly compressible* approximation,
switched on by setting `minires.ResSim.ct` ($c_t$) $> 0$.

### Definition

The **compressibility** of anything (rock or fluid) is the relative change of its
volume by pressure, $c = -\frac{1}{V} \frac{\partial V}{\partial p}$.
For rock (pores) it becomes $c_r = \frac{1}{\phi} \frac{\partial \phi}{\partial p}$,
while for fluids it is $c_f = \frac{1}{\rho} \frac{\partial \rho}{\partial p}$.
With two phases, the fluid in the pores is a mixture,
so that the **total compressibility** is the saturation-weighted sum
$c_t = c_r + s_w c_w + s_o c_o$.

### Approximation

The model, however, holds it as a single *constant*, `ct`,
so it is accurate to $O(c_t)$ alone -- the *slightly* of slightly compressible,
which is reasonable for liquids, but not for gas.
Thus $\rho \propto e^{c (p - p_0)}$, which the approximation retains only to
first order, $\rho \approx \rho_0 [1 + c (p - p_0)]$: a density affine in $p$.

Note that $\rho(p)$ (and thus nonlinearity in $p$) also appears through the source term (wells):
a rate fixed at the surface moves a reservoir volume $\propto 1/\rho(p)$.
Here it is approximated as a constant (the formation volume factor $B = 1$),
or, for a BHP well, as linear in $p$ (Peaceman).

### Derivation

Return to the conservation of mass (3), now with $\rho = \rho(p)$ and $\phi = \phi(p)$.
By the chain rule and the definitions (constant $c$) above, the accumulation term becomes
$$\frac{\partial (\rho \phi)}{\partial t}
= \rho \, \phi \, (c_r + c_f) \, \frac{\partial p}{\partial t} \,,$$
where $c_f$ is the compressibility of the fluid filling the pores.
Meanwhile, in the flux term,
$\nabla \cdot (\rho \mathbf{v}) = \rho \, \nabla \cdot \mathbf{v} + \mathbf{v} \cdot \nabla \rho$,
the latter term is $O(c)$ relative to the former
(since $\nabla \rho = \rho \, c \, \nabla p$ thanks to constant $c$),
and is therefore dropped, an approximation equivalent to the affine one above.
Dividing by $\rho$ and inserting Darcy's law (7),
we recover eqn. (1) except now with a time derivative:
$$\phi \, c_t \frac{\partial p}{\partial t} - \nabla \cdot \mathbf{K} \lambda(s) \, \nabla p = q \,. \tag{11}$$
Eqn. (11) is parabolic: a *diffusion* equation for pressure,
whose coefficient $\eta = \mathbf{K} \lambda / (\phi \, c_t)$
is the (pressure, or hydraulic) **diffusivity**.

The transport equation (2) needs a corresponding term.
The total velocity is no longer divergence-free: by eqn. (11) and reverting Darcy's law (7),
$\nabla \cdot \mathbf{v} = q - \phi \, c_t \, \partial p / \partial t$,
so the storage must be charged to the phases.
This model does so in proportion to their saturation,
$$\phi \frac{\partial s}{\partial t} + s \, \phi \, c_t \frac{\partial p}{\partial t} + \nabla \cdot (f(s)\, \mathbf{v}) = q_w \,, \tag{12}$$
which is what makes the water and oil equations sum to eqn. (11),
so that e.g. depleting a fully water-saturated reservoir leaves $s = 1$,
rather than conjuring oil out of the produced volume.
(Deriving each phase equation individually would instead charge the water
$s \, (c_r + c_w) \, \phi \, \partial p / \partial t$;
the two coincide iff $c_w = c_o$, the difference being within the $O(c_t)$ fidelity anyway.)
Ref `minires.ResSim.storage_rate`.
Both new terms vanish for $c_t = 0$, recovering eqns. (1) and (2) exactly.

### Consequences

The now parabolic pressure equation (11)
is discretized here by backward Euler over the same $\Delta t$ as the saturation step,
which adds $\phi \, c_t \, h^2 / \Delta t$ to the diagonal of the system (10),
rendering it nonsingular without pinning.
Thus **the solution method survives.**
The sequential splitting remains applicable,
and the pressure step is still *one* sparse linear solve,
with no Newton iteration on $p$, and no PVT properties
($\rho$, $\mu$, $B$, $\phi$) to update with the pressure.

- The absolute pressure level is meaningful, so an initial pressure must be given.
  However, the datum remains arbitrary. Eqn. (11) involves $p$ only through its derivatives,
  so shifting `P0` (and any BHP targets) by a constant shifts the whole pressure
  trajectory by it, leaving saturations and rates untouched. The level is thus
  *consequential* (unlike for $c_t = 0$, it is propagated, not free) but only
  relative to the initial one. An absolute pressure would enter only through
  pressure-dependent properties -- precisely what the approximation drops.
- Sources and sinks need not balance.
  The imbalance -- the **voidage**, production minus injection --
  is supplied by expansion, permitting *primary depletion* by a lone producer.
  Summing the rows of the system (ref `tests/test_compressible.py`) yields
  $c_t \, \Delta \bar{p} = V_{\text{voidage}} / V_{\text{pore}}$,
  so the fidelity requirement, $c_t \, \Delta p \ll 1$,
  is a matter of the voidage asked of the fluids, not of choosing `ct` small.
  Linearity again: the mean pressure declines in proportion to the *cumulative*
  voidage, whatever its distribution in space or time -- the straight line of the
  material-balance plot (ref `minires.ResSim.ct`), by which pore volume is estimated.
- Pressure is *transient* rather than instantaneous:
  $\sqrt{\eta t}$ is the *radius of investigation*, how far a well has "felt" after time $t$.
  Flow is called **transient** while that radius is still growing,
  and **pseudo-steady state** (or boundary-dominated) once it has reached
  the whole of the drainage volume, whereafter the pressure declines uniformly.
  **Well testing** is the inverse problem of inferring $\mathbf{K}$ and the skin
  (ref `minires.wells.peaceman_WI`) from a measured transient,
  typically during the *build-up* after shutting a well in -- as `examples.buildup` does.
  By the linear approximations, **superposition holds**, in space and in time: a shut-in is a flowing well plus an
  equal and opposite one started at the shut-in, and the pressure anywhere is the
  sum of the wells' individual transients. This is what makes **well testing** an
  *analytical* inference method -- the line-source solution, the Horner plot, and the
  semilog-derivative plateau that `examples.buildup` reads $\mathbf{K}$ off,
  are all solutions of the *linear* diffusion equation.

<!-- markdownlint-configure-file
{
  "heading-increment": false,
  "emphasis-style": false,
  "no-inline-html": {
    "allowed_elements": [ "img", "sup" ]
  },
  "ul-indent": { "indent": 2 }
}
-->

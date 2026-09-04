As mentioned in the [**the main README**](https://github.com/patnr/TPFA-ResSim) this is a
2D, two-phase, black-oil, immiscible
reservoir simulator, neglecting capillary forces and gravity,
using TPFA (two-point flux approximation),
equipped with explicit and implicit ode solvers.
By default it is incompressible, but slight compressibility
can be enabled via the `ResSim.ct` attribute.

<img src="https://github.com/patnr/TPFA-ResSim/raw/main/collage.jpg" width="100%"/>

## Governing equations

[1]: https://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

The simulator solves eqn. (1) and (2)
(corresponding to (42) and (43) of the [reference paper][1]) :

$$\phi \, c_t \frac{\partial p}{\partial t}
- \nabla \cdot \mathbf{K} \lambda(s) \, \nabla p = q \,, \tag{1}$$
$$\; \phi \frac{\partial s}{\partial t} + s \, \phi \, c_t \frac{\partial p}{\partial t}
+ \nabla \cdot (f(s)\, \mathbf{v}) = \frac{q_w}{\rho_w} \,. \tag{2}$$

Here, $c_t \geq 0$ is a constant total (rock + fluids) compressibility,
in the so-called "slightly compressible" approximation
(the reference paper treats only the incompressible case).
The default, $c_t = 0$, recovers the incompressible model,
where eqn. (1) is elliptic: pressure propagates infinitely fast,
is only defined up to an additive constant,
and sources/sinks must balance, $\sum q = 0$.
With $c_t > 0$, eqn. (1) is parabolic (a diffusion equation for pressure,
discretized here by backward Euler), the absolute pressure level is meaningful,
and injection need not balance production (storage absorbs the difference),
permitting primary depletion.

.. note:: The middle term of eqn. (2) is the $O(c_t)$ counterpart of eqn. (1)'s first.

    With $c_t > 0$ the total velocity is no longer
    divergence-free, $\nabla \cdot \mathbf{v} = q - \phi \, c_t \, \partial p /
    \partial t$, so the storage must be charged to the phases — here in
    proportion to their saturation, which is what makes the water and oil
    equations sum to eqn. (1). It vanishes for $c_t = 0$, recovering the
    reference paper's transport equation exactly.
    Ref `TPFA_ResSim.ResSim.ct` and `TPFA_ResSim.ResSim.storage_rate`.

The quantities involved are all 2D-spatial fields, namely

- $\phi \in [0, 1]$ is the porosity
- $s \in [0, 1]$ is the water saturation
- $p$ is the pressure
- $v$ is the (volumetric) flow velocity ($\mathbf{v} = \mathbf{v}_o + \mathbf{v}_w$).
- $q$ is the sources/sinks
- $\rho$ is the density
- $\lambda(s)$ is the total mobility (sum of mobilities).
  Each (relative) mobility is the phase relative permeability
  divided by the phase viscosity, $\lambda_{\text{phase}} = k_{\text{phase}}/\mu_{\text{phase}}$.
- $f(s) = \lambda_w / \lambda \in [0, 1]$ is the water fractional flow,
  where both mobilities depend on $s$.
  It gives $\mathbf{v}_w = f(s) \, \mathbf{v}$.

The right hand side of (2) is further simplified (relabelled) as $q$,
i.e. dropping the $w$ (for "water") subscripts.

.. note:: Relative permeabilities are a constituent relation, not data.

    $k_{\text{phase}} \in [0, 1]$ are set via a relation
    that is a function of the (reducible) saturation.
    They do not generally sum to 1.
    Their approximation and uncertainty is significant,
    but usually less important than those of the absolute permeability, $\mathbf{K}$.


## Derivation

#### Single phase

Conservation of mass in a (single-phase)
fluid setting is usually derived along with the concept
of "material derivative", i.e. $\frac{D}{D t}$.
This would be a bit trickier in the case of a porous medium,
but the same concepts lead to

$$\frac{∂(\rho \phi)}{∂t} + ∇ \cdot (\rho \mathbf{v}) = q \,. \tag{3}$$

This equation is also called continuity eqn., advection eqn., transport eqn.,
or even 1st-order wave eqn. (if constant $v$).
Notice that it "simply" says that divergence/convergence must be balanced
by change in density or porosity, or sinks or sources.
If we assume constant porosity, $\phi$, and incompressibility (constant $\rho$),
then the time derivative vanishes, yielding the steady-state equation
$$\nabla \cdot \mathbf{\mathbf{v}} = \frac{q}{\rho} \,. \tag{4}$$

This still leaves us with one equation for 3 unknowns.
The system is closed by *Darcy*'s law:
$$\mathbf{v} = − \frac{\mathbf{K}}{\mu} \nabla u \,, \tag{5}$$
where
$$u = p - \rho g z $$
is the *velocity potential*,
Darcy's law provides us with 3 additional equations and 1 additional unknown, $p$.
It is analogous to Fourier's heat diffusion and Ohm's conduction law,
but contains *two* forces (pressure gradient and gravity).

Darcy's law (5) may be derived from Navier-Stokes momentum equations,
but was empirically derived by Darcy.
Indeed, it is simply a formula for the velocity,
as the gradient of the potential, $u$,
linearly transformed by the permeability tensor (matrix).
Inserting the formula (5) into eqn. (4) yields
$$− \nabla \cdot \frac{\mathbf{K}}{\mu} \nabla u = \frac{q}{\rho} \,. \tag{6}$$
which can be solved for $u$.
In reservoir engineering, *no-flow* boundary conditions are most often used.
Still, $u$ is only determined up to a constant (as behoves a *potential*).
Finally, $u$ can be inserted in Darcy's law (5) to yield the (steady-state) velocity.

#### Two phases

The [reference paper](1) explains how to apply the continuity equation (3)
and Darcy's law (5) for each phase in a multiphase (and even multicomponent) flow system.
Even the black-oil case involves 27 unknowns and equations.
By assuming immiscibility and incompressibility,
and constituent relations, and astute combination of the equations,
they arrive at eqn. (1) and (2).

Here we shall be more heuristic but brief.

- Incompressibility again yields eqn. (4) for the *total* (volumetric) velocity.
- Darcy's law (5) is assumed for each (both) individual phase,
  meaning that $\mathbf{K}$ is replaced by $\mathbf{K} \lambda_{\text{phase}}(s)$.
- Neglecting $\nabla z$ (gravity, i.e. hydrostatic pressure),
  the flow potential, $u$, reduces to the pressure field, $p$.
- Summing Darcy's law over the two phases yields
  $$\mathbf{v} = − \mathbf{K} \lambda (s) \nabla p \,. \tag{7}$$
- Hence, repeating the derivation for eqn. (6), we obtain eqn. (1).

Meanwhile, conservation of mass (3)
for a *single*, incompressible phase is obtained by
replacing the density $\rho$ in eqn. (3)
by $s_{\text{phase}} \, \rho_{\text{phase}}$,
and $\mathbf{v}$ by $\mathbf{v}_{\text{phase}} = f_{\text{phase}}(s)\, \mathbf{v}$.
This immediately yields eqn. (2).


## How to solve

Equations (1) and (2) are nonlinearly coupled:
$s$ and $p$ (yielding $v$ via eqn. (7)) appear in both equations.
Trying to solve both equations simultaneously is a nonlinear root-finding problem,
requiring Newton iterations and matrix inversions.
Given this complication, it is then possible to use *implicit* time discretization
(like ECLIPSE 100) where $s_{t+1}$ is expressed as a (nonlinear) function of itself,
which also requires iterative solution.

Here, instead, we apply sequential operator splitting,
meaning that the two equations are solved independently,
inserting the previous solution of (1) into (2), and vice-versa.
Since it yields smaller systems (which can potentially be discretized explicitly)
this is faster, but less accurate.
The simulator code contains both an implicit and explicit (upwind) time discretization
for the nearly-hyperbolic saturation equation.
When using the explicit one, the strategy is called IMPES
(implicit pressure, explicit saturation), although I'm not sure why,
since the pressure equation itself does not contain a time derivative
(with $s$ fixed, equation (1) is a nearly-elliptic
boundary value problem for the pressure, $p$).

The spatial discretization is carried out by finite volumes (FV),
which is similar to finite differences (FD),
but arguably easier to formulate for non-structured (irregular) grids.
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

## Missing features

- The model is 2D
  - No gravity effects (buoyancy, drainage)
  - No coning
  - No layering
  - No vertical cross-section
- Fluids
  - No gas
  - Constant viscosity
  - No bubble point, no $R_s$, gas injection or WAG
  - Immiscible, single-component (no compositional) and no polymer, surfactant, foam, or CO₂
  - Isothermal
  - No capillary pressure
  - Rel-perm is hard-coded
- Grid is uniform, rectangular
  - No corner-point or unstructured (PEBI) geometry,
  - no faults or transmissibility multipliers
  - no local grid refinement, and no adaptivity.
- Permeability is diagonal, not a full tensor, as indicated by "TPFA"
- The boundary is no-flow, always. No aquifer model
- No geostatistics, and no upscaling
- Wells
  - No wellbore hydrostatics, friction
  - No control-mode switching, natively, outside of `TPFA_ResSim.ResSim.well_controls`
  - Wells snap to cell centres (ref `TPFA_ResSim.wells.Wells.xy`, and the note below)
- Numerics
  - First-order upwind in space, in both saturation schemes: the front is
    smeared by numerical diffusion. No higher-order, TVD or flux-limited scheme,
    no discontinuous Galerkin, and no streamline/front-tracking method.
  - No I/O: no deck (`.DATA`) parsing, no restart or summary files, no
    interoperability with the industry formats

.. note:: Spreading a well over its neighbouring cells was implemented, then omitted.

    Snapped to a cell centre, a well has its every effect a staircase function
    of its coordinates -- constant within a cell -- which leaves an
    optimisation over well *positions* with no gradient to work with until the
    well crosses into the next cell (as in the EnOpt of
    [HistoryMatching](https://github.com/patnr/HistoryMatching)). Distributing
    it over the 4 surrounding cells by bilinear weights -- a *mollified* rather
    than a rounded delta -- fixes that, and is not merely cosmetic: the
    position dependence so obtained tracks that of a 3-times-refined grid to
    within that grid's own discretization spread, i.e. some 5 times closer than
    rounding manages. It does require the well index to be corrected for the
    cells dividing the load, Peaceman's equivalent radius being derived for the
    whole source in one cell: a well divided 4 ways under-reports its drawdown
    by 23%, non-convergently, unless the weighted geometric mean of the
    intercell distances is substituted for $r_e$ (which recovers 0.3%).

    It was nonetheless judged not to earn its complexity -- a stencil threaded
    through the well assembly, the well model and the plotting, for a
    convenience of the optimiser rather than a fidelity of the simulator. The
    work is preserved on the branch `well-spread` (`ResSim.spread_wells`,
    `Grid2D.xy2stencil`, `wells._share_WI`, and `tests/test_spread.py`, which
    pins each of the measurements quoted above).

.. note:: What remains is nonetheless enough to exhibit most of what makes
    reservoir simulation interesting: the nonlinear pressure-saturation
    coupling, the hyperbolic front and its (fingering, channelling) instability,
    the drive mechanisms of injection and of depletion, the sub-grid nature of a
    well, and the disparity of scales that makes all of it a hard inverse
    problem.

## Vocabulary of reservoir engineering

Reservoir simulators implement porous media flow
on upscaled geophysical parameters typically with grid blocks between 1 - 100 m.
They usually parameterize multiphase flow.
If only the two phases of oil and water are used it is called **black-oil**.
A common assumption is that the flow is **immiscible**: not mixing (oil and water).
But this does not mean that gas cannot be *dissolved* in oil.

Fossil fuel hydrocarbons is sedimented, pressurized, organic material (mostly plants?)
that used to live on the **sub-sea** continental shelves
**On-land** organic material turns into coal.
⇒ Saudi-Arabia used to be sub-sea?
The *energy* in oil & gas comes from the sun (photosynthesis),
not the compression.

The lightest *hydrocarbons* (methane, ethane, etc.) usually escapes quickly,
while oils moves slowly towards the surface.
Sometimes the geology is bends to form caps of non-permeable rock,
so that the migrating hydrocarbons are trapped.
Upon drilling, unless valves are in place, the pressure of the initial
*equilibrium* will cause a *blow out*.
A new equilibrium is usually attained when 20% of the hydrocarbons
have been produced, which marks the end of the *primary production*.
In the *North Sea*, these reservoirs lie 1000-3000 meters below the sea bed.
Norway is also surrounded by the *Norwegian sea*,
and the *Barents sea*, towards Murmansk.

**Porosity**, $\phi$, is the *void volume fraction*.
Depends on pressure, because rock is compressible.
*Compressibility* is the porosity's (relative) gradient wrt. pressure.
Usually neglected, so that $\phi$ is a constant, but spatial, field.

**Permeability**, denoted by tensor $\mathbf{K}$, quantifies transmissibility.
Usually SPD, and correlated with $\phi$.
Among the reservoir rocks,
*sandstone* usually have large, well-connected pores,
and high permeability, *shale* is nearly impermeable,
like cap rock and bed rock.
Permeability is measured in Darcy ($≈ 10^{-12} m^2$).
A medium is called *isotropic* if $\mathbf{K}$ is scalar.

The **phases** (rock, oil, gas), whose saturations sum to $1$,
contains *components* (e.g. methane, ethane, propane),
usually grouped as pseudo-components.
Each phase's *mass fraction* component, $c_{phase,i}$, sums to $1$.
Each phase has **density**, $\rho$ and **viscosity**, $\mu$,
generally functions of the phase **pressure**,
but usually neglected except for gas.
The differences in pressure are named **capillary pressure**
because they arise due to **interfacial tensions**.
A phase's **compressibility** is defined similar as for the rock's.
Confusingly, it is also denoted with $c$, but using only a single subscript.

Phases do not really mix. But in macro-scale modelling all phases
may be present at the same location. Therefore a phase's permeability
should depend on the saturations, to which end we introduce *relative permeability*,
$k_{r,i} = k_{r,i}(s_g, s_o), i = g, o, w$
a nonlinear function, yielding an (effective) permeability
$\mathbf{K_i} = \mathbf{K} k_{r,i}$
Relative permeability curves do not extend all over the interval $[0, 1]$.
The smallest saturation where a phase is mobile is called the **residual saturation**.
This *adsorption* effects may vary, and this may have important effects,
particularly for simulation of *polymer injection*.
The uncertainty regarding relative permeability is modest compared to
the enormous uncertainty of the rock permeability.

Everything depends on *thermodynamics*, but this is often complex and neglected,
except perhaps for the bubble/boiling point pressures,
which govern how much of the gas dissolves in oil.

**Aquifers** are beneficial in reservoirs as they act as pressure compensators.
Oil production ⇒ pressure decrease ⇒ aquifers expansion ⇒ pressure compensation.
Despite consisting of water, the expansion is generally significant
because the base volume is so big,
or the aquifer might even be connected to the ocean.

#### Compressibility

The **compressibility** of anything (rock or fluid) is the relative change of its
volume per unit change of pressure, $c = -\frac{1}{V} \frac{\partial V}{\partial p}$,
equivalently $+\frac{1}{\rho} \frac{\partial \rho}{\partial p}$ for a fluid.
For the rock it is the *pore* volume that is meant, so that $c_r$ is the
(relative) gradient of the porosity, as already stated above.
The **total compressibility** of a grid block sums the rock and the (saturation
weighted) fluid contributions, $c_t = c_r + s_w c_w + s_o c_o$,
which is what `TPFA_ResSim.ResSim.ct` holds (as a single constant).
Water is around $5 \cdot 10^{-10}\, \mathrm{Pa}^{-1}$, oil a few times more,
the rock of the same order, but *gas* is around $1/p$,
i.e. two orders of magnitude larger at reservoir pressures (and more as they drop)
-- which is why the presence of free gas dominates everything.
The **slightly compressible** approximation takes $c$ to be *constant*,
so that $\rho \propto e^{c (p - p_0)} \approx \rho_0 [1 + c (p - p_0)]$;
this keeps eqn. (1) linear, and is reasonable for liquids, but not for gas.

Since compressibility relates volumes to pressure, a volume must be qualified by
where it is measured. The **formation volume factor**, $B$, is the ratio of the
volume at reservoir conditions to that of the same mass at the surface
("stock tank"), and is how field rates (measured at the surface) are converted
to the reservoir rates that a simulator works in.
This model has $B = 1$: all of its rates and volumes are reservoir ones.
Related **PVT** (pressure-volume-temperature) vocabulary:
the **bubble point** is the pressure below which gas comes out of solution;
an oil above it is **undersaturated**, and the amount of gas it holds is the
*solution gas-oil ratio*, $R_s$.

The **drive mechanism** is whatever supplies the energy that pushes the
hydrocarbons to the well. *Fluid and rock expansion* (a.k.a. **depletion drive**)
is the weakest, recovering only a few percent, because $c_t$ is so small:
this is what $c_t > 0$ enables here, in the absence of injection.
Stronger ones are *solution gas drive* (gas evolving as $p$ drops below the
bubble point), *gas cap drive*, *water drive* (the aquifers mentioned below),
*gravity drainage*, and *compaction drive* (which manifests as seabed subsidence).
Recovery is staged: **primary** production runs on the native drive;
**secondary** adds *pressure support* by injecting water or gas
(**waterflooding** being the case simulated here);
**tertiary**, or **EOR** (enhanced oil recovery), alters the flow physics itself,
e.g. by polymer, surfactant, or CO₂ injection.
The **voidage replacement ratio** is the injected reservoir volume divided by the
produced one; $\mathrm{VRR} = 1$ is exactly the balance, $\sum q = 0$,
that the incompressible model is obliged to impose.
The zero-dimensional (single tank) accounting of all of the above,
used to estimate reserves without a grid, is called **material balance**.

Compressibility is also what makes pressure *transient* rather than instantaneous.
Eqn. (1) being a diffusion equation, its coefficient
$\eta = \mathbf{K} \lambda / (\phi c_t)$ is the (pressure, or hydraulic)
**diffusivity**, and $\sqrt{\eta t}$ is the *radius of investigation*:
how far a well has "felt" after time $t$. Flow is called **transient** while that
radius is still growing, and **pseudo-steady state** (or boundary-dominated) once
it has reached the whole of the drainage volume, whereafter the pressure declines
uniformly. **Well testing** is the inverse problem of inferring $\mathbf{K}$ and
the skin (below) from a measured transient, typically during the *build-up* after
shutting a well in.

#### Wells

A well is either an **injector** or a **producer** -- the terminal states of the
sources and sinks, $q$, of the governing equations.
The two are not distinct objects here: a well is an injector or a producer
merely by the *sign* of its rate (ref `TPFA_ResSim.wells.Wells.rates`), and
under BHP control not even by that, the direction being left to the pressures
(ref `TPFA_ResSim.wells.Wells.bhp`).
Its **completion** is the equipment that connects the **wellbore** to the rock,
whose interface is the **sandface**; it may be *open hole*, or cased and
**perforated**. A well may have several completions, e.g. one per layer,
or (here) one per grid cell traversed by the well path -- which the model
assembles individually, grouping them back into wells only for the reporting
(ref `TPFA_ResSim.ResSim.wells`).

Because a wellbore (radius $r_w \sim 0.1$ m) is orders of magnitude smaller than a
grid block, its pressure is not resolved by the grid: the radial solution
$p \sim \ln r$ spends most of its variation within the well's own cell.
The **bottom-hole pressure** (BHP), $p_\mathrm{bh}$, is the pressure at the
sandface, which is thus a *sub-grid* quantity;
measured at the surface instead, it is the *tubing head pressure* (THP), the
difference being hydrostatic and friction losses in the tubing -- neither of which
a 2D areal model has, which is why BHP is where this model stops.
The **drawdown** is the pressure difference driving the flow,
$p_\mathrm{cell} - p_\mathrm{bh}$: positive for a producer (whence its name),
while for an injector it is negative, being an *overpressure*.

The **productivity index** (PI) is the resulting constant of proportionality,
$q = \mathrm{PI} \cdot \Delta p$ (the *injectivity index* for injectors),
familiar from well testing. Its counterpart in a simulator is the
**well index** (WI), which is the same relation with the fluid factored out,
$q = \mathrm{WI} \, \lambda_t \, \Delta p$,
so that $\mathrm{WI}$ depends only on geometry and rock,
and the mobility $\lambda_t$ carries the (time-varying) fluid dependence.
It is the well's analogue of the transmissibility $t_{ij}$ of eqn. (10).
Two things enter it, beyond $r_w$ and $\mathbf{K}$:

- The **skin**, $S$, is a dimensionless lumping of all *near-wellbore* effects that
  the model does not resolve. It is *positive* for damage (drilling mud invasion,
  fines migration, scale) and *negative* for stimulation (acidizing, or hydraulic
  **fracturing**), and enters additively to a logarithm, so a skin of $5$ is a lot.
- The **equivalent radius**, $r_e$, is the purely *numerical* ingredient: the radius
  at which the analytic radial pressure equals the numerical *cell* pressure,
  $r_e \approx 0.2 h$ for the 5-point stencil of TPFA. Beware that the same symbol
  is used, in well testing, for the (physical) *drainage radius*.

Ref `TPFA_ResSim.wells.peaceman_WI` for the formula combining these.

A well is **controlled** either by prescribing its rate, or its BHP,
the other then being an outcome (ref `TPFA_ResSim.wells.Wells.bhp`).
Reality is closer to the latter -- one sets a pump speed or a **choke** opening,
and the reservoir decides the rate -- but the *rate* is what is usually planned for.
Field practice is therefore rate control subject to a BHP *constraint*
(from the fracture pressure of an injector, or the lift capacity, or the bubble
point, of a producer), **switching** control mode whenever the constraint binds.
A well producing at a rate too low to be worthwhile is **shut in**;
if it cannot flow unaided it needs **artificial lift** (gas lift, or a downhole pump).

The **water cut** of a producer is the water fraction of what it produces,
i.e. the $f(s)$ of its cell; **breakthrough** is when the injected water first
arrives, after which the water cut climbs and the well eventually becomes uneconomic.
How much of the oil the flood has contacted by then is the **sweep efficiency**,
which is governed by the **mobility ratio**, $M = \lambda_w / \lambda_o$
evaluated behind and ahead of the front. $M > 1$ is *unfavourable*:
the (less viscous) water outruns the oil in **viscous fingering**,
and even more so along the high-permeability *channels* -- the motivation for
polymer injection, which fixes $M$ by thickening the water.
The corresponding vertical phenomenon (which a 2D areal model cannot see)
is **coning**, of water up, or gas down, into the completion.

Wells are drilled in repeated **patterns**, of which the *five-spot* (a producer
at the centre of four injectors, or vice-versa if *inverted*) is the classic;
by symmetry it suffices to simulate the *quarter five-spot*, as in the examples here.
They need not be vertical: *deviated*, *horizontal* and *multilateral* wells
contact more rock per well, at the price of an **allocation** problem,
namely how the total rate distributes itself among the completions
(ref `TPFA_ResSim.wells.well_path`).
Later interventions to restore or improve a well are **workovers**,
and drilling extra wells between the existing ones is **infill drilling**.

Other lingo:
water table, facies, channels, fissures, fractures.

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

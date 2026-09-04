TPFA-ResSim is a
2D, two-phase, black-oil, immiscible
reservoir simulator
using TPFA (two-point flux approximation).

The `examples` (whose figures make up the collage) have pages of their own here.

## Governing equations

[1]: https://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

The simulator solves eqn. (1) and (2)
(corresponding to (42) and (43) of the [reference paper][1]) :

$$- \nabla \cdot \mathbf{K} \lambda(s) \, \nabla p = q \,, \tag{1}$$
$$\; \phi \frac{\partial s}{\partial t}
+ \nabla \cdot (f(s)\, \mathbf{v}) = \frac{q_w}{\rho_w} \,. \tag{2}$$

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

These are the equations of the *incompressible* model,
which is the default here. Compressibility is treated further below,
as an extension of the solution method.


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
*Darcy's law* provides 2 additional equations and 1 additional unknown, pressure $p$:
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
In reservoir engineering, *no-flow* boundary conditions are most often used,
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
- Meanwhile, conservation of mass (3)
  for a *single*, incompressible phase is obtained by
  replacing the density $\rho$ in eqn. (3)
  by $s_{\text{phase}} \, \rho_{\text{phase}}$,
  and $\mathbf{v}$ by $\mathbf{v}_{\text{phase}} = f_{\text{phase}}(s)\, \mathbf{v}$.
  This immediately yields eqn. (2).
  It is here that *immiscibility* enters: each phase occupies its own share of the
  pore volume ($s_w + s_o = 1$) and conserves its own mass, so that eqn. (3) holds
  per phase, with no exchange term between them
  (and hence no dependence of $\rho$ or $\mu$ on composition).


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
The simulator contains both an implicit and explicit (upwind) time discretization
for the nearly-hyperbolic saturation equation.
The implicit scheme for the saturation equation scheme does not appear to earn its keep,
however, ref `ResSim.saturation_step_implicit`.
When using the explicit one, the strategy is called IMPES
(implicit pressure, explicit saturation).

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

    Its matrix changes only through the mobility $λ(s)$ (and, with $c_t > 0$,
    the accumulation term), so the factorization of an earlier step is an
    excellent **preconditioner** for the current one, at the cost of
    a back-substitution, and a refactorization only once the iteration stalls.
    `tests/test_precond.py` benchmarks.

## Compressibility

The above is the incompressible model, which is what the reference paper treats,
and the default here. Optionally, the simulator adds compressibility,
in the so-called *slightly compressible* approximation,
switched on by setting `TPFA_ResSim.ResSim.ct` ($c_t$) $> 0$.

### Definition

The **compressibility** of anything (rock or fluid) is the relative change of its
volume per unit change of pressure, $c = -\frac{1}{V} \frac{\partial V}{\partial p}$,
equivalently $+\frac{1}{\rho} \frac{\partial \rho}{\partial p}$ for a fluid.
For the rock it is the *pore* volume that is meant,
so that $c_r = \frac{1}{\phi} \frac{\partial \phi}{\partial p}$.
Water is around $5 \cdot 10^{-10}\, \mathrm{Pa}^{-1}$, oil a few times more,
the rock of the same order, but *gas* is around $1/p$,
i.e. two orders of magnitude larger at reservoir pressures (and more as they drop)
-- which is why the presence of free gas dominates everything.
The **slightly compressible** approximation takes $c$ to be *constant*,
so that $\rho \propto e^{c (p - p_0)} \approx \rho_0 [1 + c (p - p_0)]$;
this keeps the pressure equation linear,
and is reasonable for liquids, but not for gas.

### Derivation

Return to the conservation of mass (3), now with $\rho = \rho(p)$ and $\phi = \phi(p)$.
By the chain rule and the definitions above, the accumulation term becomes
$$\frac{\partial (\rho \phi)}{\partial t}
= \rho \, \phi \, (c_r + c_f) \, \frac{\partial p}{\partial t} \,,$$
where $c_f$ is the compressibility of the fluid filling the pores.
In the flux term, $\nabla \cdot (\rho \mathbf{v})
= \rho \, \nabla \cdot \mathbf{v} + \mathbf{v} \cdot \nabla \rho$,
the latter is $O(c)$ relative to the former (since $\nabla \rho = \rho \, c \, \nabla p$),
and is dropped -- as is the pressure dependence of $\rho$ in the wells,
so that reservoir and surface volumes are not distinguished ($B = 1$, see below).
Dividing by $\rho$ and inserting Darcy's law (7), eqn. (1) acquires a time derivative:
$$\phi \, c_t \frac{\partial p}{\partial t}
- \nabla \cdot \mathbf{K} \lambda(s) \, \nabla p = q \,. \tag{11}$$
With two phases, the fluid in the pores is a mixture,
so that the **total compressibility** is the saturation-weighted sum
$c_t = c_r + s_w c_w + s_o c_o$.
The model, however, holds it as a single *constant*, `ct`,
so it is accurate to $O(c_t)$ alone -- the *slightly* of slightly compressible.

The transport equation (2) needs a corresponding term.
The total velocity is no longer divergence-free: by eqn. (11),
$\nabla \cdot \mathbf{v} = q - \phi \, c_t \, \partial p / \partial t$,
so the storage must be charged to the phases.
This model does so in proportion to their saturation,
$$\phi \frac{\partial s}{\partial t} + s \, \phi \, c_t \frac{\partial p}{\partial t}
+ \nabla \cdot (f(s)\, \mathbf{v}) = q_w \,, \tag{12}$$
which is what makes the water and oil equations sum to eqn. (11),
so that e.g. depleting a fully water-saturated reservoir leaves $s = 1$,
rather than conjuring oil out of the produced volume.
(Deriving each phase equation individually would instead charge the water
$s \, (c_r + c_w) \, \phi \, \partial p / \partial t$;
the two coincide iff $c_w = c_o$, the difference being within the $O(c_t)$ fidelity anyway.)
Ref `TPFA_ResSim.ResSim.storage_rate`.
Both new terms vanish for $c_t = 0$, recovering eqns. (1) and (2) exactly.

### Consequences

Eqn. (11) is parabolic: a *diffusion* equation for pressure,
whose coefficient $\eta = \mathbf{K} \lambda / (\phi \, c_t)$
is the (pressure, or hydraulic) **diffusivity**.
It is discretized here by backward Euler over the same $\Delta t$ as the saturation step,
which adds $\phi \, c_t \, h^2 / \Delta t$ to the diagonal of the system (10),
rendering it nonsingular without the pinning of the first element.
Compared to the incompressible model:

- The absolute pressure level is meaningful, so an initial pressure must be given.
- Sources and sinks need not balance.
  The imbalance -- the **voidage**, production minus injection --
  is supplied by expansion, permitting *primary depletion* by a lone producer.
  Summing the rows of the system (ref `tests/test_compressible.py`) yields
  $c_t \, \Delta \bar{p} = V_{\text{voidage}} / V_{\text{pore}}$,
  so the fidelity requirement, $c_t \, \Delta p \ll 1$,
  is a matter of the voidage asked of the fluids, not of choosing `ct` small.
- Pressure is *transient* rather than instantaneous:
  $\sqrt{\eta t}$ is the *radius of investigation*, how far a well has "felt" after time $t$.
  Flow is called **transient** while that radius is still growing,
  and **pseudo-steady state** (or boundary-dominated) once it has reached
  the whole of the drainage volume, whereafter the pressure declines uniformly.
  **Well testing** is the inverse problem of inferring $\mathbf{K}$ and the skin
  (ref `TPFA_ResSim.wells.peaceman_WI`) from a measured transient,
  typically during the *build-up* after shutting a well in -- as `examples.buildup` does.

### Vocabulary

Since compressibility relates volumes to pressure,
a volume must be qualified by where it is measured.
The **formation volume factor**, $B$, is the ratio of the volume at reservoir conditions
to that of the same mass at the surface ("stock tank"),
and is how field rates (measured at the surface) are converted
to the reservoir rates that a simulator works in. This model has $B = 1$.
Related **PVT** (pressure-volume-temperature) vocabulary:
the **bubble point** is the pressure below which gas comes out of solution;
an oil above it is **undersaturated**, and the amount of gas it holds is the
*solution gas-oil ratio*, $R_s$.

The **drive mechanism** is whatever supplies the energy that pushes the
hydrocarbons to the well. *Fluid and rock expansion* (a.k.a. **depletion drive**),
which is what $c_t > 0$ enables here in the absence of injection,
is the weakest, recovering only a few percent, because $c_t$ is so small.
Stronger ones are *solution gas drive*, *gas cap drive*, *water drive* (aquifers),
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

## Sensitivities

`TPFA_ResSim.tlm` is the hand-derived adjoint of a time step with respect to
the *state*, $ (s, p) $, and to $ \log K $: `tlm.linearize` recomputes a step
(from the trajectory that `ResSim.sim` returns) into a `tlm.Tape`, and
`tlm.adj_step` propagates a sensitivity back through it. Along a trajectory,
`tlm.adjoint` turns the partials of an objective with respect to the stored
states into its gradient with respect to `S0`, `P0` and $ \log K $, at the cost
of about one more simulation. The step's tangent is a straight line of
sparse-matrix and diagonal statements around one symmetric solve, and the
adjoint is its reversal, statement by statement -- verified against finite
differences of the objective in `tests/test_tlm.py`. See
`examples.water_cut_gradient` for the gradient of one producer's water cut
with respect to the $ \log K $ field, and `examples.history_match_gradient`
for that of a production-history misfit, put to use in a few descent steps.

## Missing features, and alternatives

The model is 2D and areal: no gravity (buoyancy, drainage), no coning, layering or
vertical cross-section. The fluids are two, immiscible and isothermal -- water and oil of
constant viscosity -- with no gas (no bubble point, $R_s$, gas injection or WAG), no
compositional, polymer, surfactant, foam or CO₂ modelling, no capillary pressure, and
hard-coded relative permeabilities. The grid is uniform and rectangular: no corner-point
or unstructured (PEBI) geometry, no faults or transmissibility multipliers, no local
refinement or adaptivity; and the permeability is diagonal, not a full tensor, as "TPFA"
indicates. The boundary is always no-flow, without an aquifer model, and there is neither
geostatistics nor upscaling. The wells have no wellbore hydrostatics or friction, no
control-mode switching natively (only via `TPFA_ResSim.ResSim.well_controls`), and snap
to cell centres (ref `TPFA_ResSim.wells.Wells.xy`, and the note below). The numerics are
first-order upwind in space, in both saturation schemes, so the front is smeared by
numerical diffusion -- no higher-order, TVD or flux-limited scheme, no discontinuous
Galerkin, no streamline or front-tracking method. And there is no I/O: no deck (`.DATA`)
parsing, no restart or summary files, no interoperability with the industry formats.

.. note:: What remains is nonetheless enough to exhibit most of what makes
    reservoir simulation interesting: the nonlinear pressure-saturation
    coupling, the hyperbolic front and its (fingering, channelling) instability,
    the drive mechanisms of injection and of depletion, the sub-grid nature of a
    well, and the disparity of scales that makes all of it a hard inverse
    problem.

Nor is there much else to choose from. This simulator exists because, when the
[HistoryMatching](https://github.com/patnr/HistoryMatching) tutorial needed a forward
model, there was no Python reservoir simulator that was at once *simple* (readable in a
sitting), *two-phase* (fronts and breakthrough being the instructive part), *reliable*
(verified against a published reference, tested in CI), *importable* as a plain object for
ensemble loops, and permissively licensed. A survey in August 2026 found that combination
still unoccupied: searching GitHub for Python reservoir simulation turns up ensemble
*tooling* ([ERT](https://github.com/equinor/ert), [pyscal](https://github.com/equinor/pyscal))
and abandoned toolboxes -- the scarcity is on the forward-model side.

### Simple, teaching-oriented simulators

| Project | Physics | Status (Aug 2026) | Remarks |
|---|---|---|---|
| [chanshing/ressim](https://github.com/chanshing/ressim) | 2D two-phase, source terms only | dead since 2019 | Nearest sibling, from the *same* Aarnes, Gimse & Lie paper; but no well API, no verification against the Matlab original, no tests or packaging |
| [dageo](https://github.com/tuda-geo/dageo) (ex-*resmda*) | 2D **single-phase** | active, on PyPI/conda | Data-assimilation teaching package (ES-MDA, localization, random fields) with a pressure-only toy model attached, so no saturation fronts. Complements rather than replaces this one; its docs invite external simulators |
| [mashadab/Reservoir-Simulator](https://github.com/mashadab/Reservoir-Simulator) | 1D/2D, two-phase, wells, gravity | inactive since 2022 | Benchmarked against CMG, but course-homework structure, no license, no tests |
| [reservoirflow](https://github.com/hiesabx/reservoirflow) | single-phase so far | beta | Ambitious, pip-installable, but "most modules not fully functional"; CC BY-NC-SA |
| [pyMRST](https://github.com/yohanesnuwara/pyMRST) | wraps MRST via Octave | maintained | Real install friction, subprocess and file I/O per call; GPL |

### Full-featured simulators

Reliable, but not simple, and not readable end-to-end by a student:
[open-DARTS](https://open-darts.readthedocs.io) (C++ core with a Python API, and the
biggest change since 2022 in being a genuine `pip install`),
[OPM Flow](https://opm-project.org) (industrial black-oil, deck-driven, invoked as a
subprocess per realization by ensemble tools such as ERT),
[JutulDarcy.jl](https://github.com/sintefmath/JutulDarcy.jl) (arguably the best open-source
simulator today -- SPE-validated, MRST-compatible, differentiable -- but Julia), and
MRST itself (Matlab, the original motivation).

At scale these are vastly faster: compiled kernels, AMG/CPR iterative solvers, parallelism.
At tutorial scale ($10^3$--$10^4$ cells) those advantages are irrelevant and their fixed
overheads dominate -- process launch, deck generation and parsing, result-file I/O, package
loading and JIT warm-up, engine initialization -- while a direct sparse solve on a small 2D
system is near-optimal. So a simulator that runs in-process as a plain Python object, with
no files or handshakes, is plausibly as fast or faster wall-clock for the thousands of short
runs an ensemble tutorial performs. That is reasoning; the following is measurement.

### Measured against JutulDarcy.jl

Three of JutulDarcy's own examples fall inside this simulator's envelope (2D areal, two
immiscible phases, quadratic Corey, rate/BHP wells, weak compressibility) and were posed
identically in both codes: **Buckley--Leverett** (1D, 25--800 cells), the **quarter
five-spot** of `five_spot_ensemble.jl` (rate injector, BHP producer, grid and report steps
refined together from $25^2$ to $200^2$), and **SPE10 model 2, layer 1** (60x220, the
injector on rate under a BHP limit that binds throughout, 25--400 report steps).

A fully implicit Newton code and a sequential explicit-transport code do not do comparable
work per step, so the comparison is of wall time at *equal RMS error of the produced water
cut* -- against the analytic Welge forecast for Buckley--Leverett, otherwise against each
code's own finest run -- with grid and report steps as the knobs. Single thread, warm (JIT
and package loading excluded), JutulDarcy 0.3.10 at default tolerances and "relaxed" by one
order (`tol_cnv 1e-2`, `tol_mb 1e-6`), TPFA-ResSim at commit `d061a41`. The codes agree:
finest runs within 0.003--0.005 RMS of each other.

| case | RMS error | TPFA-ResSim | JutulDarcy default | JutulDarcy relaxed |
|---|---|---|---|---|
| Buckley--Leverett | 0.08 | 0.021 s | 0.14 s | 0.10 s |
| Buckley--Leverett | 0.05 | 0.17 s | 1.3 s | 1.0 s |
| Quarter five-spot | 0.012 | 2.4 s | 0.98 s | 0.90 s |
| Quarter five-spot | 0.008 | 10.7 s | 8.2 s | 7.5 s |
| Quarter five-spot | 0.006 | 28 s | 23 s | 18 s |
| SPE10 layer 1 | 0.004 | 3.3 s | 5.5 s | 4.4 s |
| SPE10 layer 1 | 0.0015 | 5.2 s | 9.6 s | 6.9 s |

- **Buckley--Leverett: TPFA-ResSim 6--7x faster.** Its explicit upwind transport smears
  the shock over fewer cells than the fully implicit scheme, so at equal grid it is both
  faster and more accurate.
- **Quarter five-spot: JutulDarcy 1.2--2.4x faster.** At equal *grid* TPFA-ResSim is
  faster (4--5x at $50^2$, parity at $400^2$: its cost grows as $\sim N^{3.5}$ from the
  CFL sub-steps, JutulDarcy's as $\sim N^3$), but its error at equal grid is 1.3--1.7x
  larger, and with error falling only as $\sim t^{-0.3}$ in either code, buying that
  accuracy back costs a factor 2--3 in time.
- **SPE10 layer 1: TPFA-ResSim 1.4--1.8x faster.** JutulDarcy's temporal error per report
  step is 2--3x smaller (the sequential splitting lags pressure one step behind
  saturation), but each of its steps costs 2x more.
- Relaxing JutulDarcy's tolerances saves 10--25% at no measurable accuracy cost; its
  defaults are conservative, and the relaxed column is the fairer one to quote.

So "as fast or faster at tutorial scale" holds for the small grids and short runs of an
ensemble tutorial, but not as a rule: on a 2D flood at equal accuracy the fully implicit
code is the more efficient from $\sim 50^2$ up -- and the comparison excludes everything
JutulDarcy has that this simulator lacks (3D, gravity, black-oil, general grids,
parallelism). The scripts, per-run curves and write-up are in `benchmarks/jutul/` on the
branch `bench-jutul`.

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

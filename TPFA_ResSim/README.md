As mentioned in the [**the main README**](https://github.com/patnr/TPFA-ResSim) this is a
2D, two-phase, black-oil, immiscible, incompressible
reservoir simulator, neglecting capillary forces and gravity,
using TPFA (two-point flux approximation),
equipped with explicit and implicit ode solvers.

<img src="https://github.com/patnr/TPFA-ResSim/raw/main/collage.jpg" width="100%"/>

## Governing equations

[1]: https://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf

The simulator solves eqn. (1) and (2)
(corresponding to (42) and (43) of the [reference paper][1]) :

$$- \nabla \cdot \mathbf{K} \lambda(s) \, \nabla p = q \,, \tag{1}$$
$$\; \phi \frac{\partial s}{\partial t} + \nabla \cdot (f(s)\, \mathbf{v})
= \frac{q_w}{\rho_w} \,. \tag{2}$$

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

.. note:: Relative permeabilities, $k_{\text{phase}} \in [0, 1]$,
    are set via a constituent relation that is a function of the (reducible) saturation.
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

The spatial discretization is carried out by finite volumes (FV).
For the pressure equation, it is known as two-point flux approximation (TPFA),
using only two points two approximate the transmissibility and fluxes at the
interfaces; simple, but used widely (nearly default) in oil industry, due to
its robustness and efficiency.
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
water table, facies, channels, water cut, fissures, fractures.

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

# vel-pid-demo

Simulations to understand the differences between the positional and velocity
(incremental) forms of the PID controller, and how the reference PID
implementation of Sundström et al. (2024) compares with both.

## Goals

- Show that the positional and velocity forms give the same control
  performance when the controller gains are fixed.
- Show what happens to the controller output when the gains are changed while
  the loop is running (adaptive gains, e.g. switching between heating and
  cooling gains).
- Find out which of these behaviours the reference PID implementation has,
  when it is configured to match the other two as closely as possible.

## The three controllers

| Name | Source | Implementation |
| --- | --- | --- |
| Positional | Rockwell white paper [1], page 2 | `rockwell_pid.PositionalPID` |
| Velocity | Rockwell white paper [1], page 2 | `rockwell_pid.VelocityPID` |
| Reference | Sundström et al. (2024) [2] | `python_pid.PIDController` from [pid-ref](https://github.com/billtubbs/pid-ref) |

The `rockwell_pid` package implements the two equations in the white paper
directly. It is not an implementation of the PIDE instruction itself, which is
a velocity form with more features.

### Positional form

$$CV = K_P E + \sum K_I E \Delta t + K_D \frac{\Delta E}{\Delta t}$$

The proportional term works on the error, and the integral term is accumulated
in a separate sum. When the output is limited, the integral sum is
recalculated so that the output equals the limit (anti-windup).

### Velocity form

$$CV_n = CV_{n-1} + K_P \Delta E + K_I E \Delta t + K_D \frac{E_n - 2E_{n-1} + E_{n-2}}{\Delta t}$$

The proportional term works on the change in error, and the integral
accumulation is contained in the previous output $CV_{n-1}$. The limited
output is stored as $CV_{n-1}$, which gives the velocity form its built-in
anti-windup.

Block diagrams of both forms are in the docstrings of
[positional.py](src/rockwell_pid/positional.py) and
[velocity.py](src/rockwell_pid/velocity.py).

### Reference PID

The reference PID is also a velocity form, but the proportional and derivative
increments are calculated from the stored values of those terms at the
previous step:

$$\Delta u_P = k_p (b r - y_f) - u_{P,\text{old}}$$

where $y_f$ is the filtered measurement and $b$ the setpoint weight. It also
filters the measurement, takes the derivative of the filtered measurement
rather than the error, and supports feedforward, manual and tracking modes
and a variable execution period.

In the demo it is wrapped in `ReferencePID` and configured to match the other
two as closely as possible:

| Setting | Value | Why |
| --- | --- | --- |
| `kp`, `ki`, `kd` | $K_P$, $K_I \Delta t$, $K_D / \Delta t$ | It uses gains per sample period |
| `b` | 1 | Proportional term acts on the error |
| `u0` | 0 | No bias |
| `umin`, `umax` | −100, 100 | Same output limits |
| `TfTs` | 0.001 | Switches the measurement filter off |

`TfTs = 0` isn't allowed (it divides by zero), but with `TfTs = 0.001` the
discretized filter coefficients are exactly $b_1 = 1$ and all others 0 in
floating point, so $y_f = y$. This also makes the filtered derivative exactly
0, which switches off the derivative term, so this configuration is only
suitable with $K_D = 0$, which all the scenarios use.

How the filter should behave with $T_f = 0$ was raised by the reference
implementation's authors in issue
[copybit/PID.code#2](https://github.com/copybit/PID.code/issues/2) ("Filter
behaviour when Tf=0"). It names two options: pass the measurement through
unfiltered and approximate the derivative with a simple finite difference, or
pass it through unfiltered and output a derivative of 0, which disables the
D part. `TfTs = 0.001` gives the second option.

The difference that remains is the stored proportional term. With PI control
and the filter off, this makes the reference PID identical to the Rockwell
positional form (see [below](#the-reference-pid-is-the-positional-form-computed-incrementally)).

## Scenarios and results

All three controllers control a first-order-plus-dead-time process
($K = 1$, $\tau = 10$ s, $\theta = 2$ s, $\Delta t = 0.1$ s), with PI gains from
lambda tuning ($\lambda = \theta$): $K_P = 2.5$, $K_I = 0.25$.

| Scenario | Positional | Velocity | Reference |
| --- | --- | --- | --- |
| 1. Setpoint changes, fixed gains: max output difference from Positional | 0 | 0 | 0 |
| 2. Gains × 1.5 at steady state ($E \approx 0$): output step | 0.002 | 0 | 0.002 |
| 3. Gains × 2 with $E \approx 5$: output step | 12.0 | 0 * | 12.0 |

\* The velocity form's output changes by −1.0 in that time step, which is just
its normal change in output over one step.

Scenario 3 is the situation described on pages 4–5 of the white paper: the
gains are changed while there is an error of about 5.

### Findings

1. With fixed gains all three give the same response. The positional and
   velocity forms are mathematically equivalent.
2. When $K_P$ changes, the positional form steps its output by
   $\Delta K_P \times E$. There is no step at steady state, but there is one
   whenever there is an error.
3. The velocity form never steps its output when the gains change, because the
   proportional term works on the change in error.
4. In both Rockwell forms $K_I$ multiplies each integral increment, so a change
   in $K_I$ only affects future integral action.
5. **On a change in $k_p$, the reference PID behaves like the positional form,
   not the Rockwell velocity form**: its output steps by $\Delta K_P \times E$.
   Its output matches the positional form's to within $10^{-13}$ throughout
   scenarios 2 and 3.

## Bumpless gain changes

The Rockwell velocity form is the bumpless one. Its source is the Rockwell
white paper [1]: the equation on page 2, and the example on page 5 ("Changing
the proportional gain from 1 to 3 with a 5% error causes no change in the
output since the error was 5% just before the new gain was used and remained
5% just after the new gain was used"). The `VelocityPID` class is unchanged
from the first version of this demo.

It differs from the reference PID's velocity form only in the proportional
(and derivative) increment:

| | Rockwell velocity form | Reference PID |
| --- | --- | --- |
| Proportional increment | $K_P (E_n - E_{n-1})$ | $k_{p,n} E_n - k_{p,n-1} E_{n-1}$ (stored term) |
| Output step when $K_P$ changes | 0 | $\Delta K_P \times E_n$ |
| Output after a gain change | The effect of the old $K_P$ stays in $CV_{n-1}$ | Always current P and D terms + accumulated integral |

The reference PID's choice is a trade-off rather than a bug. Its output always
decomposes into the current proportional and derivative terms plus the
accumulated integral, so a new $k_p$ takes full effect immediately. That is
also what lets it add the bias `u0` for P and PD control. The cost is a step of
$\Delta K_P \times E$ when $k_p$ changes while there is an error.

The newer implementation in [copybit/PID.code](https://github.com/copybit/PID.code)
[3] (April 2026, from the GitHub account that published the reference
pseudo-code) uses the bumpless increment instead. In `python/PID.py`
the proportional increment is computed from the change in setpoint and
measurement, `du_p = kp * (b * d_r - d_y)`, and the derivative increment in
the same way, so a gain change doesn't step the output. It also replaces the
inherent anti-windup with back-calculation (time constant `Tt`). This
repository doesn't include that version.

## The reference PID is the positional form computed incrementally

"Velocity form" describes how the output is computed, $CV_n = CV_{n-1} +
\Delta CV_n$. How the output responds to a gain change depends on how the
proportional increment is defined. Summing the increments shows the
difference. Take PI control with $b = 1$, no filter, no output limiting and
$CV_0 = 0$, and add up the increments from step 1 to $n$.

**Reference PID.** The proportional increment is
$k_{p,k} E_k - k_{p,k-1} E_{k-1}$, so the sum telescopes: each stored term
cancels the next one, and only the current term is left:

$$CV_n = \sum_{k=1}^{n} \left( k_{p,k} E_k - k_{p,k-1} E_{k-1} \right) + \sum_{k=1}^{n} k_{i,k} E_k = k_{p,n} E_n + \sum_{k=1}^{n} k_{i,k} E_k$$

(taking $E_0 = 0$). This is the Rockwell positional form,
$CV = K_P E + \sum K_I E \Delta t$, with $k_i = K_I \Delta t$, and it holds
even when the gains change. So a change in $k_p$ steps the output by
$\Delta K_P \times E_n$, exactly as in the positional form.

**Rockwell velocity form.** The proportional increment is
$K_{P,k} (E_k - E_{k-1})$, with the gain in force at step $k$, so the sum does
not telescope:

$$CV_n = \sum_{k=1}^{n} K_{P,k} (E_k - E_{k-1}) + \sum_{k=1}^{n} K_{I,k} E_k \Delta t$$

The proportional part is a running sum of error changes, each weighted by the
gain at the time. A change in $K_P$ only affects future error changes, so the
output doesn't step, but afterwards it no longer equals current $K_P \times E$
plus the integral.

With constant gains both sums reduce to $K_P E_n + \sum K_I E \Delta t$,
which is why all three controllers are identical in scenario 1.

Output limiting doesn't separate the reference PID from the positional form
either: the reference PID stores the limited output as $CV_{n-1}$, and
`PositionalPID` recalculates its integral sum so that $P + I + D$ equals the
limit, which leaves both in the same state.

So the reference PID uses the velocity form for its implementation benefits
(the integral is held in $CV_{n-1}$, which gives inherent anti-windup,
bumpless switching from manual and tracking modes, and a variable sample
period) but keeps positional-form behaviour on gain changes.

## Note on the integral term

An earlier version of `PositionalPID` calculated the integral term as
$K_I \sum E \Delta t$, with $K_I$ outside the sum. This is common in practice,
for example in the dependent (ISA) form, where $K_C / T_I$ multiplies the
whole integral. With fixed gains it gives the same result, but a change in
$K_I$ rescales the whole accumulated integral and steps the output by a further
$\Delta K_I \times \int E \, dt$, which is large even at steady state. The
white paper puts $K_I$ inside the sum, so the class now does too.

## Installation and use

Requires Python 3.9 or later.

```sh
pip install -r requirements.txt
python vel_pid_demo.py
```

`requirements.txt` installs the reference PID from GitHub and this repo's
`rockwell_pid` package in editable mode. To use a local copy of pid-ref
instead, install it with `pip install -e ../pid-ref`.

The script prints the results and saves plots to `plots/`.

## References

1. Rockwell Automation, *Perform Common Process Loop Control Algorithms Using
   the PIDE Instruction*, white paper Logix-WP008.
   <https://literature.rockwellautomation.com/idc/groups/literature/documents/wp/logix-wp008_-en-p.pdf>
2. E. Sundström, T. Hägglund, M. Bauer, J. Eker, K. Soltesz, *Reference
   Implementation of the PID Controller*, IFAC-PapersOnLine, 58(7), 2024,
   pp. 370–375. <https://doi.org/10.1016/j.ifacol.2024.08.090>
   Pseudo-code: <https://github.com/copybit/pid>
3. copybit, *PID.code*: MATLAB and Python implementations of the PID
   controller, with examples. <https://github.com/copybit/PID.code>

"""
Demonstration of Velocity Form vs Positional Form PID Controllers

Compares three PID controllers on a First Order Plus Dead Time (FOPDT)
process, typical of many industrial processes like temperature control:

1. Positional form, as in Rockwell Automation's white paper
   "Perform Common Process Loop Control Algorithms Using the PIDE
   Instruction" (Logix-WP008), page 2:
       CV = Kp*E + sum(Ki*E*Δt) + Kd*(E_n - E_{n-1})/Δt
2. Velocity form, from the same page:
       CV_n = CV_{n-1} + Kp*(E_n - E_{n-1}) + Ki*E_n*Δt
              + Kd*(E_n - 2*E_{n-1} + E_{n-2})/Δt
3. Reference PID of Sundström et al. (2024), a velocity (incremental)
   form, using the Python implementation python_pid.PIDController from
   https://github.com/billtubbs/pid-ref. It is configured to match the
   Rockwell forms as closely as its parameters allow (see ReferencePID).

Key demonstrations:
1. Under normal operation (fixed gains) all three give the same
   response
2. When the gains change, the positional form steps its output by
   ΔKp × E because its proportional term works on the error directly.
   At steady state (E = 0) there is no bump.
3. The Rockwell velocity form works on the change in error, so a gain
   change never bumps the output. The reference PID stores the previous
   proportional term kp*(b*r - yf) rather than the previous error, so on
   a change in kp it behaves like the positional form: the output steps
   by ΔKp × E. In fact, summing its increments shows that it is the
   positional form computed incrementally (see README.md)

Plots are saved to the plots/ sub-folder.
"""

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from python_pid import PIDController

from rockwell_pid import PositionalPID, VelocityPID

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Measurement filter time constant of the reference PID, as a multiple of
# the sample time (Tf = TF_TS * dt). The reference PID always filters the
# measurement with a second-order filter (default TfTs = 10) and the
# Rockwell forms have no filter. TfTs = 0 is not allowed (division by
# zero), but with TfTs = 0.001 the discretized filter coefficients are
# exactly a11 = a12 = a21 = a22 = b2 = 0 and b1 = 1 in floating point, so
# yf = y: the filter is switched off. It also sets the filtered
# derivative dyf to 0, which switches off the derivative term, so this
# setting is only suitable with Kd = 0.
TF_TS = 0.001


class ReferencePID:
    """
    Reference PID controller (Sundström et al., 2024)

    Wraps python_pid.PIDController with the same interface as the
    Rockwell controllers, configured to match them as closely as
    possible:

    - kp = Kp, ki = Ki * Δt, kd = Kd / Δt (it uses per-sample gains
      with a normalized execution period Tx = 1)
    - Setpoint weight b = 1, so the P term acts on the error
    - Bias u0 = 0 and the same output limits
    - Measurement filter switched off with TfTs = TF_TS
    - No feedforward, manual, tracking or external windup signals

    Control signal increments (with Tx = 1, b = 1):

        ΔCV = kp*(r - yf) - up_old + ki*(r - yf) + (-kd*dyf - ud_old)
        CV_n = CV_{n-1} + ΔCV

    where yf and dyf are the filtered measurement and its derivative, and
    up_old, ud_old are the proportional and derivative terms from the
    previous step. The integral accumulation is contained in CV_{n-1}.

    Differences that remain from the Rockwell velocity form:
    - P increment is kp_n*E_n - kp_{n-1}*E_{n-1} rather than
      Kp*(E_n - E_{n-1}), so a change in kp steps the output by ΔKp × E
      (like the positional form)
    - Derivative acts on the filtered measurement, not the error
      (and is switched off by TF_TS, see above)
    """

    def __init__(self, Kp, Ki, Kd, dt, output_limits=(-100, 100), TfTs=TF_TS):
        self.dt = dt
        self.pid = PIDController(
            kp=Kp,
            ki=Ki * dt,
            kd=Kd / dt,
            TfTs=TfTs,
            umin=output_limits[0],
            umax=output_limits[1],
            u0=0.0,
            b=1.0,
        )

    def update(self, setpoint, pv):
        """Compute PID output using the reference velocity form"""
        return self.pid(setpoint, pv)

    def set_gains(self, Kp=None, Ki=None, Kd=None):
        """Change controller gains (no bump from the integral term)"""
        if Kp is not None:
            self.pid.kp = Kp
        if Ki is not None:
            self.pid.ki = Ki * self.dt
        if Kd is not None:
            self.pid.kd = Kd / self.dt


# Controllers compared in each scenario, with plot line styles
CONTROLLERS = {
    "Positional": (PositionalPID, "-"),
    "Velocity": (VelocityPID, "--"),
    "Reference": (ReferencePID, "-."),
}


class FOPDTProcess:
    """
    First Order Plus Dead Time (FOPDT) Process Model

    Transfer function: G(s) = K * e^(-θs) / (τs + 1)

    Parameters:
        K = process gain (steady-state gain)
        θ (theta) = dead time (transport delay)
        τ (tau) = time constant

    This model is representative of many thermal processes like temperature control,
    solar collectors, and the heat/cool examples in the Rockwell document.
    """

    def __init__(self, K=1.0, tau=10.0, theta=2.0, dt=0.1):
        self.K = K  # Process gain
        self.tau = tau  # Time constant (seconds)
        self.theta = theta  # Dead time (seconds)
        self.dt = dt  # Sample time

        # State variables
        self.pv = 0.0  # Current process variable

        # Dead time buffer (FIFO queue)
        self.delay_steps = int(theta / dt)
        self.delay_buffer = deque(
            [0.0] * self.delay_steps, maxlen=self.delay_steps
        )

    def update(self, u):
        """
        Update process model using discrete first-order equation with dead time

        Discrete first-order: PV_{n+1} = PV_n + (dt/τ) * (K*u_delayed - PV_n)
        """
        # Get delayed input
        u_delayed = self.delay_buffer[0] if self.delay_buffer else u

        # Add current input to delay buffer
        self.delay_buffer.append(u)

        # First-order response to delayed input
        self.pv = self.pv + (self.dt / self.tau) * (
            self.K * u_delayed - self.pv
        )

        return self.pv


def simulate_scenario(
    process,
    controller,
    setpoint_profile,
    gain_change_time=None,
    new_gains=None,
    duration=100,
):
    """Run a simulation scenario and return time-series data"""
    dt = process.dt
    n_steps = int(duration / dt)

    # Storage arrays
    time = np.zeros(n_steps)
    setpoints = np.zeros(n_steps)
    pvs = np.zeros(n_steps)
    outputs = np.zeros(n_steps)
    errors = np.zeros(n_steps)

    # Setpoint generator
    setpoint_idx = 0
    current_setpoint = setpoint_profile[0][1]

    for i in range(n_steps):
        t = i * dt
        time[i] = t

        # Update setpoint based on profile
        if setpoint_idx < len(setpoint_profile) - 1:
            if t >= setpoint_profile[setpoint_idx + 1][0]:
                setpoint_idx += 1
                current_setpoint = setpoint_profile[setpoint_idx][1]

        setpoints[i] = current_setpoint

        # Change gains if specified
        if (
            gain_change_time is not None
            and gain_change_time <= t < gain_change_time + dt
        ):
            controller.set_gains(**new_gains)

        # PID control
        error = current_setpoint - process.pv  # error seen by controller
        output = controller.update(current_setpoint, process.pv)

        # Process update
        pv = process.update(output)

        # Store results
        pvs[i] = pv
        outputs[i] = output
        errors[i] = error

    return {
        "time": time,
        "setpoint": setpoints,
        "pv": pvs,
        "output": outputs,
        "error": errors,
    }


def run_controllers(
    process_params,
    gains,
    dt,
    setpoint_profile,
    gain_change_time=None,
    new_gains=None,
    duration=100,
):
    """Simulate every controller in CONTROLLERS on the same scenario.

    Returns a dict of results keyed by controller name.
    """
    results = {}
    for name, (controller_class, _) in CONTROLLERS.items():
        process = FOPDTProcess(**process_params, dt=dt)
        controller = controller_class(*gains, dt)
        results[name] = simulate_scenario(
            process,
            controller,
            setpoint_profile,
            gain_change_time=gain_change_time,
            new_gains=new_gains,
            duration=duration,
        )
    return results


def make_ts_plot(results_dict, title, filename=None):
    """Plot time series of simulation results from several controllers.

    Subplots: process variable and setpoint, control error, and
    controller output. Returns fig, axes.
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True)

    def linestyle(name):
        return CONTROLLERS[name][1] if name in CONTROLLERS else "-"

    ax = axes[0]
    for name, results in results_dict.items():
        ax.plot(
            results["time"],
            results["pv"],
            linestyle(name),
            label=name,
            alpha=0.8,
        )
    first = next(iter(results_dict.values()))
    ax.plot(first["time"], first["setpoint"], "k:", label="Setpoint")
    ax.set_ylabel("PV")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_title("Process variable")

    ax = axes[1]
    for name, results in results_dict.items():
        ax.plot(
            results["time"],
            results["error"],
            linestyle(name),
            label=name,
            alpha=0.8,
        )
    ax.set_ylabel("Error")
    ax.grid(True)
    ax.set_title("Control error")

    ax = axes[2]
    for name, results in results_dict.items():
        ax.plot(
            results["time"],
            results["output"],
            linestyle(name),
            label=name,
            alpha=0.8,
        )
    ax.set_ylabel("CV")
    ax.set_xlabel("Time (s)")
    ax.grid(True)
    ax.set_title("Controller output")

    fig.suptitle(title)
    plt.tight_layout()

    if filename:
        PLOTS_DIR.mkdir(exist_ok=True)
        filepath = PLOTS_DIR / filename
        fig.savefig(filepath, dpi=150)
        print(f"\nPlot saved: {PLOTS_DIR.name}/{filename}")

    return fig, axes


def output_step(results, idx):
    """Change in controller output at time step idx"""
    return results["output"][idx] - results["output"][idx - 1]


def print_gain_change_analysis(results, idx, d_Kp):
    """Print the output step at a gain change for each controller,
    and the part of it theoretically caused by the gain change."""
    # Error seen by the controllers at the gain change (practically the
    # same for all three before the change)
    error = results["Positional"]["error"][idx]
    print(f"\n  Error at gain change:  E = {error:.4f}")

    theory = {
        "Positional": ("ΔKp × E", d_Kp * error),
        "Velocity": ("0", 0.0),
        "Reference": ("ΔKp × E", d_Kp * error),
    }
    print("\n  Output step at gain change:")
    print(f"    {'Controller':<12}{'Simulated':>12}   Due to gain change")
    for name in CONTROLLERS:
        expr, value = theory[name]
        print(
            f"    {name:<12}{output_step(results[name], idx):>12.4f}"
            f"   {expr} = {value:.4f}"
        )
    print("  (The simulated step also includes the normal change in output")
    print("  over one time step.)")


def main():
    """Main demonstration script"""

    print("=" * 80)
    print("PID Controller Comparison: Velocity Form vs Positional Form")
    print("=" * 80)
    print("\nPositional: positional form (Rockwell white paper Logix-WP008)")
    print("Velocity:   velocity form (Rockwell white paper Logix-WP008)")
    print("Reference:  reference PID of Sundström et al. (2024)")
    print(f"            (measurement filter TfTs = {TF_TS})")
    print("\nProcess Model: First Order Plus Dead Time (FOPDT)")
    print("  - Process Gain (K): 1.0")
    print("  - Time Constant (τ): 10.0 seconds")
    print("  - Dead Time (θ): 2.0 seconds")
    print("  - Sample Time (Δt): 0.1 seconds")
    print("\n" + "=" * 80)

    # Simulation parameters
    dt = 0.1  # Sample time

    # PID tuning using lambda tuning method for FOPDT
    # For FOPDT: Kp = τ/(K*(λ+θ)), Ti = τ, Td = 0
    # Using λ = θ (common choice for good robustness)
    K_process = 1.0
    tau = 10.0
    theta = 2.0
    lambda_tune = theta
    process_params = {"K": K_process, "tau": tau, "theta": theta}

    Kp = tau / (K_process * (lambda_tune + theta))
    Ti = tau
    Ki = Kp / Ti  # Convert to gain form
    Kd = 0.0  # Keep it simple - no derivative action
    gains = (Kp, Ki, Kd)

    print("\nInitial PID Tuning (Lambda Tuning Method):")
    print(f"  Kp = {Kp:.3f}")
    print(f"  Ki = {Ki:.3f}")
    print(f"  Kd = {Kd:.3f}")

    # ===================================================================
    # SCENARIO 1: Normal Operation - All Forms (Practically) Identical
    # ===================================================================
    print("\n" + "-" * 80)
    print("SCENARIO 1: Normal Operation (Setpoint Changes)")
    print("-" * 80)
    print("All three should give practically identical responses")

    setpoint_profile_1 = [(0, 0), (10, 10), (40, 15), (70, 5)]
    results_1 = run_controllers(process_params, gains, dt, setpoint_profile_1)

    print("\nPerformance Metrics:")
    print(f"  {'Controller':<12}{'IAE':>10}{'Max |ΔCV| vs Positional':>27}")
    for name, results in results_1.items():
        iae = np.sum(np.abs(results["error"])) * dt
        max_diff = np.max(
            np.abs(results["output"] - results_1["Positional"]["output"])
        )
        print(f"  {name:<12}{iae:>10.4f}{max_diff:>27.6f}")

    make_ts_plot(
        results_1,
        "Scenario 1: Normal Operation",
        "scenario1_normal_operation.png",
    )

    # ===================================================================
    # SCENARIO 2: Adaptive Gains at Steady State
    # ===================================================================
    print("\n" + "-" * 80)
    print("SCENARIO 2: Adaptive Gains at Steady State")
    print("-" * 80)
    print("Simulating heat/cool process where gains change")
    print("(like extrusion machine example in Rockwell document)")

    setpoint_profile_2 = [
        (0, 0),
        (10, 10),
        (50, 10),  # Hold at setpoint when gains change
    ]

    # New gains for "cooling mode" (more aggressive)
    new_Kp = Kp * 1.5
    new_Ki = Ki * 1.5
    gain_change_time = 50.0

    print(f"\nGain change at t={gain_change_time}s (process at steady state):")
    print(f"  Old Kp = {Kp:.3f}, New Kp = {new_Kp:.3f}")
    print(f"  Old Ki = {Ki:.3f}, New Ki = {new_Ki:.3f}")

    results_2 = run_controllers(
        process_params,
        gains,
        dt,
        setpoint_profile_2,
        gain_change_time=gain_change_time,
        new_gains={"Kp": new_Kp, "Ki": new_Ki},
    )
    print_gain_change_analysis(
        results_2, int(gain_change_time / dt), new_Kp - Kp
    )

    make_ts_plot(
        results_2,
        "Scenario 2: Gain Change at Steady State",
        "scenario2_adaptive_gains.png",
    )

    # ===================================================================
    # SCENARIO 3: Gain Change With Error - Key Advantage of Velocity Form
    # ===================================================================
    print("\n" + "-" * 80)
    print("SCENARIO 3: Gain Change With Error (Key Advantage of Velocity Form)")
    print("-" * 80)
    print("Gain change while the error is still large after a setpoint change")
    print("(like the 5% error example on pages 4-5 of the Rockwell document)")

    setpoint_profile_3 = [
        (0, 0),
        (10, 10),
        (30, 20),  # Large setpoint change
    ]

    # More aggressive gains (tripling them makes the loop oscillate)
    new_Kp_3 = Kp * 2.0
    new_Ki_3 = Ki * 2.0
    gain_change_time_3 = 34.0  # Error is about 5

    print(f"\nGain change at t={gain_change_time_3}s:")
    print(f"  Old Kp = {Kp:.3f}, New Kp = {new_Kp_3:.3f}")
    print(f"  Old Ki = {Ki:.3f}, New Ki = {new_Ki_3:.3f}")

    results_3 = run_controllers(
        process_params,
        gains,
        dt,
        setpoint_profile_3,
        gain_change_time=gain_change_time_3,
        new_gains={"Kp": new_Kp_3, "Ki": new_Ki_3},
    )
    print_gain_change_analysis(
        results_3, int(round(gain_change_time_3 / dt)), new_Kp_3 - Kp
    )

    make_ts_plot(
        results_3,
        "Scenario 3: Gain Change With Error",
        "scenario3_large_error.png",
    )

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 80)
    print("""
1. SAME PERFORMANCE UNDER NORMAL CONDITIONS:
   With fixed gains, the positional and velocity forms are mathematically
   equivalent, and all three give the same response.

2. GAIN CHANGES: THE PROPORTIONAL TERM MAKES THE DIFFERENCE
   - Positional: the P term works on the error, Kp × E, so a gain change
     steps the output by ΔKp × E. No bump at steady state (E = 0), but a
     large one if there is an error.
   - Rockwell velocity form: the P term works on the change in error,
     Kp × (E_n - E_{n-1}), so a gain change never bumps the output.
   - In both Rockwell forms Ki is applied to each increment, Ki × E × Δt, so a
     change in Ki only affects future integral action.

3. THE REFERENCE PID BEHAVES LIKE THE POSITIONAL FORM ON A KP CHANGE
   - It stores the previous P term, kp × (b × r - yf), not the previous
     error, so a change in kp also steps the output by ΔKp × E
   - Summing its increments, the stored P terms cancel and leave
     CV_n = kp_n × E_n + sum(ki × E): the positional form, computed
     incrementally
   - The trade-off: the new kp takes effect immediately and the output
     always equals current P + D terms + accumulated integral, which is
     also what lets it add the bias u0 in P and PD control. In the
     Rockwell velocity form the effect of the old Kp stays in CV_{n-1}.

4. PRACTICAL IMPLICATIONS (from Rockwell document):
   - Velocity form is used in most DCS systems
   - Enables easy implementation of adaptive gains
   - Simplifies multiloop selection and cascade control
   - Much easier to program and maintain
    """)

    print(f"\nAll plots saved to {PLOTS_DIR.name}/")
    print("\nFor more details, see:")
    print("  Rockwell Automation Logix-WP008")
    print(
        "  'Perform Common Process Loop Control Algorithms Using the PIDE Instruction'"
    )
    print("  Sundström et al. (2024), Reference Implementation of the PID")
    print("  Controller, https://doi.org/10.1016/j.ifacol.2024.08.090")
    print("=" * 80)

    plt.show()


if __name__ == "__main__":
    main()

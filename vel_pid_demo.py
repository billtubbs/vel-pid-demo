"""
Demonstration of Velocity Form vs Positional Form PID Controllers

Compares a textbook positional-form PID controller with the reference
velocity-form (incremental) PID controller of Sundström et al. (2024),
using the Python implementation in the pid-ref repository
(https://github.com/billtubbs/pid-ref).

The scenarios follow Rockwell Automation's PIDE instruction documentation
(Logix-WP008B-EN-P). Both controllers control a First Order Plus Dead Time
(FOPDT) process, which is typical of many industrial processes like
temperature control.

Key demonstrations:
1. Under normal operation (fixed gains) both forms give practically the
   same response
2. When the gains change, the positional form bumps its output because
   its integral term Ki*sum(E*dt) is rescaled by the new Ki
3. The velocity form keeps the accumulated integral action in the
   previous output, so a gain change does not bump it. The reference
   implementation stores the previous proportional term kp*(b*r - yf)
   rather than the previous error, so a change in kp still gives a small
   bump of size Delta-kp * E (zero when the error is zero)

Plots are saved to the plots/ sub-folder.
"""

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from python_pid import PIDController

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Measurement filter time constant of the reference PID, as a multiple of
# the sample time (Tf = TF_TS * dt). The reference PID always filters the
# measurement with a second-order filter (default TfTs = 10). A small
# value is used here so that the filter has a negligible effect and the
# comparison isolates the difference between the two PID forms.
TF_TS = 0.1


class PositionalPID:
    """
    Positional Form PID Controller

    From Rockwell documentation (page 2):
    CV = Kp*E + Ki*sum(E*Δt) + Kd*(E - E_{n-1})/Δt

    The integral term accumulates as a summation.
    The proportional term works on error directly.
    """

    def __init__(self, Kp, Ki, Kd, dt, output_limits=(-100, 100)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limits = output_limits

        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.output = 0.0

    def update(self, setpoint, pv):
        """Compute PID output using positional form"""
        error = setpoint - pv

        # Proportional term (works on error directly)
        P = self.Kp * error

        # Integral term (accumulates in separate variable)
        self.integral += error * self.dt
        I = self.Ki * self.integral

        # Derivative term (works on change in error)
        D = self.Kd * (error - self.prev_error) / self.dt

        # Compute output
        self.output = P + I + D

        # Apply output limits
        self.output = np.clip(self.output, *self.output_limits)

        # Anti-windup: prevent integral from accumulating beyond limits
        if (
            self.output == self.output_limits[0]
            or self.output == self.output_limits[1]
        ):
            self.integral = (
                (self.output - P - D) / self.Ki if self.Ki != 0 else 0
            )

        self.prev_error = error

        return self.output

    def set_gains(self, Kp=None, Ki=None, Kd=None):
        """Change controller gains (will cause bump if error exists)"""
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd


class ReferencePID:
    """
    Velocity Form PID Controller (reference implementation)

    Wraps python_pid.PIDController (Sundström et al., 2024) with the same
    interface as PositionalPID, converting the continuous-time gains to
    the per-sample gains it uses:

        kp = Kp,  ki = Ki * Δt,  kd = Kd / Δt

    Control signal increments (with Tx = 1, setpoint weight b = 1):

        ΔCV = kp*(r - yf) - up_old + ki*(r - yf) + (-kd*dyf - ud_old)
        CV_n = CV_{n-1} + ΔCV

    where yf and dyf are the filtered measurement and its derivative, and
    up_old, ud_old are the proportional and derivative terms from the
    previous step. The integral accumulation is contained in CV_{n-1}.
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
    scenario_name,
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
            print(f"  {scenario_name}: Gains changed at t={t:.1f}s")

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


def make_ts_plot(results_dict, title, filename=None):
    """Plot time series of simulation results from several controllers.

    Subplots: process variable and setpoint, control error, and
    controller output. Returns fig, axes.
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True)

    ax = axes[0]
    for name, results in results_dict.items():
        ax.plot(results["time"], results["pv"], label=name, alpha=0.8)
    setpoint = next(iter(results_dict.values()))["setpoint"]
    time = next(iter(results_dict.values()))["time"]
    ax.plot(time, setpoint, "k:", label="Setpoint")
    ax.set_ylabel("PV")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_title("Process variable")

    ax = axes[1]
    for name, results in results_dict.items():
        ax.plot(results["time"], results["error"], label=name, alpha=0.8)
    ax.set_ylabel("Error")
    ax.grid(True)
    ax.set_title("Control error")

    ax = axes[2]
    for name, results in results_dict.items():
        ax.plot(results["time"], results["output"], label=name, alpha=0.8)
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


def output_bump(results, idx):
    """Change in controller output at time step idx"""
    return results["output"][idx] - results["output"][idx - 1]


def main():
    """Main demonstration script"""

    print("=" * 80)
    print("PID Controller Comparison: Velocity Form vs Positional Form")
    print("=" * 80)
    print("\nPositional form: textbook (Rockwell PIDE documentation)")
    print("Velocity form:   reference PID of Sundström et al. (2024)")
    print(f"                 (measurement filter TfTs = {TF_TS})")
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

    Kp = tau / (K_process * (lambda_tune + theta))
    Ti = tau
    Ki = Kp / Ti  # Convert to gain form
    Kd = 0.0  # Keep it simple - no derivative action

    print("\nInitial PID Tuning (Lambda Tuning Method):")
    print(f"  Kp = {Kp:.3f}")
    print(f"  Ki = {Ki:.3f}")
    print(f"  Kd = {Kd:.3f}")

    # ===================================================================
    # SCENARIO 1: Normal Operation - Both Forms (Practically) Identical
    # ===================================================================
    print("\n" + "-" * 80)
    print("SCENARIO 1: Normal Operation (Setpoint Changes)")
    print("-" * 80)
    print("Both forms should give practically identical responses")

    setpoint_profile_1 = [(0, 0), (10, 10), (40, 15), (70, 5)]

    # Positional form
    process_pos_1 = FOPDTProcess(K=K_process, tau=tau, theta=theta, dt=dt)
    controller_pos_1 = PositionalPID(Kp, Ki, Kd, dt)
    results_pos_1 = simulate_scenario(
        "Positional Form",
        process_pos_1,
        controller_pos_1,
        setpoint_profile_1,
        duration=100,
    )

    # Velocity form
    process_vel_1 = FOPDTProcess(K=K_process, tau=tau, theta=theta, dt=dt)
    controller_vel_1 = ReferencePID(Kp, Ki, Kd, dt)
    results_vel_1 = simulate_scenario(
        "Velocity Form",
        process_vel_1,
        controller_vel_1,
        setpoint_profile_1,
        duration=100,
    )

    # Calculate performance metrics
    iae_pos = np.sum(np.abs(results_pos_1["error"])) * dt
    iae_vel = np.sum(np.abs(results_vel_1["error"])) * dt
    max_diff = np.max(np.abs(results_pos_1["output"] - results_vel_1["output"]))

    print("\nPerformance Metrics:")
    print(f"  Positional Form IAE: {iae_pos:.4f}")
    print(f"  Velocity Form IAE:   {iae_vel:.4f}")
    print(f"  Difference:          {abs(iae_pos - iae_vel):.6f}")
    print(f"  Max difference in controller output: {max_diff:.6f}")

    make_ts_plot(
        {"Positional Form": results_pos_1, "Velocity Form": results_vel_1},
        "Scenario 1: Normal Operation",
        "scenario1_normal_operation.png",
    )

    # ===================================================================
    # SCENARIO 2: Adaptive Gains - Key Advantage of Velocity Form
    # ===================================================================
    print("\n" + "-" * 80)
    print("SCENARIO 2: Adaptive Gains (Key Advantage of Velocity Form)")
    print("-" * 80)
    print("Simulating heat/cool process where gains change")
    print("(like extrusion machine example in Rockwell document)")

    setpoint_profile_2 = [
        (0, 0),
        (10, 10),
        (50, 10),  # Hold at setpoint when gains change
    ]

    # New gains for "cooling mode" (more aggressive)
    new_Kp = Kp * 2.0
    new_Ki = Ki * 2.0
    new_gains = {"Kp": new_Kp, "Ki": new_Ki}

    gain_change_time = 50.0

    print(f"\nGain change at t={gain_change_time}s (process at steady state):")
    print(f"  Old Kp = {Kp:.3f}, New Kp = {new_Kp:.3f}")
    print(f"  Old Ki = {Ki:.3f}, New Ki = {new_Ki:.3f}")

    # Positional form with gain change
    process_pos_2 = FOPDTProcess(K=K_process, tau=tau, theta=theta, dt=dt)
    controller_pos_2 = PositionalPID(Kp, Ki, Kd, dt)
    results_pos_2 = simulate_scenario(
        "Positional Form (with gain change)",
        process_pos_2,
        controller_pos_2,
        setpoint_profile_2,
        gain_change_time=gain_change_time,
        new_gains=new_gains,
        duration=100,
    )

    # Velocity form with gain change
    process_vel_2 = FOPDTProcess(K=K_process, tau=tau, theta=theta, dt=dt)
    controller_vel_2 = ReferencePID(Kp, Ki, Kd, dt)
    results_vel_2 = simulate_scenario(
        "Velocity Form (with gain change)",
        process_vel_2,
        controller_vel_2,
        setpoint_profile_2,
        gain_change_time=gain_change_time,
        new_gains=new_gains,
        duration=100,
    )

    # Find the output bumps at the gain change
    idx = int(gain_change_time / dt)
    error = results_pos_2["error"][idx]
    integral = np.sum(results_pos_2["error"][:idx]) * dt

    print("\nOutput Bump Analysis:")
    print(f"  Error at gain change:     {error:.4f}")
    print(f"  Integral of error:        {integral:.4f}")
    print(f"  Positional Form bump:     {output_bump(results_pos_2, idx):.4f}")
    print(f"  Velocity Form bump:       {output_bump(results_vel_2, idx):.4f}")
    print("\n  Theoretical bumps:")
    print(
        f"    Positional: ΔKp × E + ΔKi × ∫E = "
        f"{(new_Kp - Kp) * error + (new_Ki - Ki) * integral:.4f}"
    )
    print(f"    Velocity:   ΔKp × E            = {(new_Kp - Kp) * error:.4f}")

    make_ts_plot(
        {"Positional Form": results_pos_2, "Velocity Form": results_vel_2},
        "Scenario 2: Adaptive Gains",
        "scenario2_adaptive_gains.png",
    )

    # ===================================================================
    # SCENARIO 3: Large Error at Gain Change
    # ===================================================================
    print("\n" + "-" * 80)
    print("SCENARIO 3: Large Error During Gain Change")
    print("-" * 80)
    print("Gain change shortly after a large setpoint change")

    setpoint_profile_3 = [
        (0, 0),
        (10, 10),
        (30, 20),  # Large setpoint change
        (50, 20),  # Change gains while tracking
    ]

    # Even more aggressive gains
    new_Kp_3 = Kp * 3.0
    new_Ki_3 = Ki * 3.0
    new_gains_3 = {"Kp": new_Kp_3, "Ki": new_Ki_3}

    gain_change_time_3 = 50.0

    # Positional form
    process_pos_3 = FOPDTProcess(K=K_process, tau=tau, theta=theta, dt=dt)
    controller_pos_3 = PositionalPID(Kp, Ki, Kd, dt)
    results_pos_3 = simulate_scenario(
        "Positional Form (large error)",
        process_pos_3,
        controller_pos_3,
        setpoint_profile_3,
        gain_change_time=gain_change_time_3,
        new_gains=new_gains_3,
        duration=100,
    )

    # Velocity form
    process_vel_3 = FOPDTProcess(K=K_process, tau=tau, theta=theta, dt=dt)
    controller_vel_3 = ReferencePID(Kp, Ki, Kd, dt)
    results_vel_3 = simulate_scenario(
        "Velocity Form (large error)",
        process_vel_3,
        controller_vel_3,
        setpoint_profile_3,
        gain_change_time=gain_change_time_3,
        new_gains=new_gains_3,
        duration=100,
    )

    idx_3 = int(gain_change_time_3 / dt)
    error_3 = results_pos_3["error"][idx_3]
    integral_3 = np.sum(results_pos_3["error"][:idx_3]) * dt

    print("\nLarge Error Scenario:")
    print(f"  Error at gain change:     {error_3:.4f}")
    print(f"  Integral of error:        {integral_3:.4f}")
    print(f"  Positional Form bump:     {output_bump(results_pos_3, idx_3):.4f}")
    print(f"  Velocity Form bump:       {output_bump(results_vel_3, idx_3):.4f}")
    print("\n  Theoretical bumps:")
    print(
        f"    Positional: ΔKp × E + ΔKi × ∫E = "
        f"{(new_Kp_3 - Kp) * error_3 + (new_Ki_3 - Ki) * integral_3:.4f}"
    )
    print(f"    Velocity:   ΔKp × E            = {(new_Kp_3 - Kp) * error_3:.4f}")

    make_ts_plot(
        {"Positional Form": results_pos_3, "Velocity Form": results_vel_3},
        "Scenario 3: Large Error During Gain Change",
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
   equivalent. Small differences here come from the reference PID's
   measurement filter (TfTs).

2. INTEGRAL TERM: WHERE THE BUMP COMES FROM
   - Positional: integral accumulated in a separate sum, I = Ki × ∫E,
     so changing Ki rescales the whole accumulated integral:
     Bump = ΔKp × E + ΔKi × ∫E (large even when E = 0)
   - Velocity: integral accumulated in the previous output (CV_{n-1}),
     so changing Ki only affects future increments

3. PROPORTIONAL TERM IN THE REFERENCE PID:
   - The reference PID stores the previous proportional term
     kp × (b × r - yf), not the previous error, so a change in kp gives
     Bump = ΔKp × E. This is zero at steady state and small when the
     error is small.
   - The textbook velocity form Kp × (E_n - E_{n-1}) has no bump at all

4. PRACTICAL IMPLICATIONS (from Rockwell document):
   - Velocity form is used in most DCS systems
   - Enables easy implementation of adaptive gains
   - Simplifies multiloop selection and cascade control
   - Much easier to program and maintain
    """)

    print(f"\nAll plots saved to {PLOTS_DIR.name}/")
    print("\nFor more details, see:")
    print("  Rockwell Automation Logix-WP008B-EN-P")
    print(
        "  'Perform Common Process Loop Control Algorithms Using the PIDE Instruction'"
    )
    print("  Sundström et al. (2024), Reference Implementation of the PID")
    print("  Controller, https://doi.org/10.1016/j.ifacol.2024.08.090")
    print("=" * 80)

    plt.show()


if __name__ == "__main__":
    main()

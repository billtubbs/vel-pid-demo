"""Positional form PID controller.

Positional form, as described in Rockwell Automation's white
paper "Perform Common Process Loop Control Algorithms Using the PIDE
Instruction" (Logix-WP008), page 2.
"""

import numpy as np


class PositionalPID:
    """
    Positional Form PID Controller

    From the Rockwell white paper (page 2):

        CV = Kp*E + sum(Ki*E*Δt) + Kd*ΔE/Δt

    with ΔE = E_n - E_{n-1}. The integral term is accumulated in a
    separate sum, I_n = I_{n-1} + Ki*E_n*Δt, and the proportional term
    works on the error directly.

    Block diagram (one execution of the loop, n = time step):

            +    E   ┌───────────────────────────┐ P + D      ┌───────┐
    SP ──▶(Σ)───┬───▶│ Kp·E_n                    │─────▶(Σ)──▶│ Limit │──┬──▶ CV
           ▲ −  │    │ + Kd·(E_n − E_{n−1})/Δt   │     + ▲    └───────┘  │
           │    │    └───────────────────────────┘       │ I_n           │
           │    │    ┌───────────────────────────┐       │               │
           │    └───▶│ I_n = I_{n−1} + Ki·E_n·Δt │───────┘               │
           │         └───────────────────────────┘                       │
           │                     ┌─────────┐                             │
           └──────── PV ─────────│ Process │◀────────────────────────────┘
                                 └─────────┘

    Because Ki is inside the sum, a change in Ki only affects future
    integral increments. A change in Kp steps the output by ΔKp × E
    (pages 4-5 of the white paper: changing Kp from 1 to 3 with a 5% error
    gives a 10% bump in output).

    Anti-windup: when the output is limited, the integral sum is
    recalculated so that P + I + D equals the limit (not shown in the
    diagram).
    """

    def __init__(self, Kp, Ki, Kd, dt, output_limits=(-100, 100)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limits = output_limits

        # State variables
        self.integral = 0.0  # I_{n-1} = sum(Ki*E*Δt)
        self.prev_error = 0.0  # E_{n-1}
        self.output = 0.0

    def update(self, setpoint, pv):
        """Compute PID output using positional form"""
        error = setpoint - pv

        # Proportional term (works on error directly)
        P = self.Kp * error

        # Integral term (accumulates in separate sum, with Ki inside it)
        self.integral += self.Ki * error * self.dt
        I = self.integral

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
            self.integral = self.output - P - D if self.Ki != 0 else 0.0

        self.prev_error = error

        return self.output

    def set_gains(self, Kp=None, Ki=None, Kd=None):
        """Change controller gains (bumps output by ΔKp × E)"""
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd

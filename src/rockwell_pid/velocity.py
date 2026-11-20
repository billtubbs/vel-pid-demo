"""Velocity form PID controller.

Velocity form, as described in Rockwell Automation's white
paper "Perform Common Process Loop Control Algorithms Using the PIDE
Instruction" (Logix-WP008), page 2.
"""

import numpy as np


class VelocityPID:
    """
    Velocity Form PID Controller

    From the Rockwell white paper (page 2):

        CV_n = CV_{n-1} + Kp*ΔE + Ki*E*Δt
               + Kd*(E_n - 2*E_{n-1} + E_{n-2})/Δt

    with ΔE = E_n - E_{n-1}. The integral accumulation is contained in
    the previous output CV_{n-1}, and the proportional term works on the
    change in error.

    Block diagram (one execution of the loop, n = time step):

            +     E   ┌────────────────────────┐ ΔCV        ┌───────┐
    SP ──▶(Σ)────────▶│ Kp·(E_n − E_{n−1})     │─────▶(Σ)──▶│ Limit │──┬──▶ CV
           ▲ −        │ + Ki·E_n·Δt            │     + ▲    └───────┘  │
           │          │ + Kd·(E_n − 2·E_{n−1}  │       │ CV_{n−1}      │
           │          │      + E_{n−2})/Δt     │       │    ┌─────┐    │
           │          └────────────────────────┘       └────│ z⁻¹ │◀───┤
           │                                                └─────┘    │
           │                     ┌─────────┐                           │
           └──────── PV ─────────│ Process │◀──────────────────────────┘
                                 └─────────┘

    z⁻¹ is a one-step delay: the limited output CV_n is stored and used
    as CV_{n-1} at the next step. Storing the limited value gives the
    velocity form its built-in anti-windup, and changing any gain does
    not bump the output (page 5 of the white paper).
    """

    def __init__(self, Kp, Ki, Kd, dt, output_limits=(-100, 100)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limits = output_limits

        # State variables
        self.output = 0.0  # CV_{n-1}
        self.prev_error = 0.0  # E_{n-1}
        self.prev_prev_error = 0.0  # E_{n-2}

    def update(self, setpoint, pv):
        """Compute PID output using velocity form"""
        error = setpoint - pv

        # Proportional term (works on CHANGE in error)
        delta_P = self.Kp * (error - self.prev_error)

        # Integral term (works on current error)
        delta_I = self.Ki * error * self.dt

        # Derivative term (works on second difference of error)
        delta_D = (
            self.Kd
            * (error - 2 * self.prev_error + self.prev_prev_error)
            / self.dt
        )

        # Compute output (previous output + changes)
        self.output = self.output + delta_P + delta_I + delta_D

        # Apply output limits
        self.output = np.clip(self.output, *self.output_limits)

        # Update error history
        self.prev_prev_error = self.prev_error
        self.prev_error = error

        return self.output

    def set_gains(self, Kp=None, Ki=None, Kd=None):
        """Change controller gains (bumpless in velocity form)"""
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd

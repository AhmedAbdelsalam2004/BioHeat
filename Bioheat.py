import numpy as np
import matplotlib.pyplot as plt


# Function to generate heat input based on type
def heat_input(t, heat_type, amplitude, frequency):
    if heat_type == "sinusoid":
        return amplitude * np.sin(2 * np.pi * t / frequency) if frequency != 0 else 0
    elif heat_type == "constant":
        return amplitude
    else:
        return 0


# Data for multiple individuals with realistic parameter values
data = [
    {
        'ID': 1,
        'rho': 1000, 'c': 3500, 'k': 0.5, 'rho_b': 1060, 'c_b': 3600, 'omega_b': 0.002,
        'T_b': 37, 'Q_m': 400, 'T0': 37, 'L': 0.1, 't_end': 300, 'dt': 1,
        'heat_type': 'sinusoid', 'amplitude': 0.05, 'frequency': 60  # Lowered amplitude
    },
    {
        'ID': 2,
        'rho': 1050, 'c': 3600, 'k': 0.45, 'rho_b': 1050, 'c_b': 3400, 'omega_b': 0.003,
        'T_b': 37, 'Q_m': 350, 'T0': 37, 'L': 0.12, 't_end': 400, 'dt': 1,
        'heat_type': 'constant', 'amplitude': 0.02, 'frequency': 0  # Lower constant heat input
    }
]

# Loop through each individual and solve the bioheat equation
for person in data:
    # Extract the parameters from the individual
    rho = person['rho']
    c = person['c']
    k = person['k']
    rho_b = person['rho_b']
    c_b = person['c_b']
    omega_b = person['omega_b']
    T_b = person['T_b']
    Q_m = person['Q_m']
    T0 = person['T0']
    L = person['L']
    t_end = person['t_end']
    dt = person['dt']
    heat_type = person['heat_type']
    amplitude = person['amplitude']
    frequency = person['frequency']

    # Simulation parameters
    Nx = 50  # Number of spatial points
    dx = L / Nx  # Spatial step size
    Nt = int(t_end / dt)  # Number of time steps

    # Stability condition (for explicit methods)
    alpha = k / (rho * c)  # Thermal diffusivity
    stability = alpha * dt / (dx ** 2)
    if stability >= 0.5:
        print(f"Warning: Stability condition not satisfied for person {person['ID']} (alpha*dt/dx^2 = {stability}).")
        continue  # Skip to the next person to avoid instability

    # Boundary conditions
    T_left = T0  # Left boundary temperature
    T_right = T0  # Right boundary temperature

    # Initial temperature distribution
    T_initial = T0 * np.ones(Nx)

    # Initialize temperature array
    T = np.copy(T_initial)
    T_new = np.copy(T)

    # Store temperature over time for visualization
    temperature_history = []

    # Time loop
    for n in range(Nt):
        # Apply boundary conditions
        T_new[0] = T_left
        T_new[-1] = T_right

        # Update temperature for internal points
        for i in range(1, Nx - 1):
            # Finite difference for spatial term (d^2T/dx^2)
            conduction = alpha * (T[i + 1] - 2 * T[i] + T[i - 1]) / (dx ** 2)
            # Blood perfusion and metabolic heat terms
            perfusion = rho_b * c_b * omega_b * (T_b - T[i]) / (rho * c)
            metabolic = Q_m / (rho * c)
            # Total change in temperature (including arbitrary heat input)
            T_new[i] = T[i] + dt * (
                        conduction + perfusion + metabolic + heat_input(n * dt, heat_type, amplitude, frequency))
            # Clamp temperature to a realistic range (35°C to 40°C for the human body)
            T_new[i] = np.clip(T_new[i], 35, 40)

        # Update temperature for the next time step
        T[:] = T_new[:]

        # Store the temperature at this time step
        temperature_history.append(np.copy(T))

    # Plot the results
    plt.figure(figsize=(8, 6))
    for i in range(0, Nt, max(1, Nt // 10)):  # Plot 10 time points
        plt.plot(np.linspace(0, L, Nx), temperature_history[i], label=f'Time = {i * dt:.0f}s')
    plt.xlabel('Position along muscle (m)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Distribution for Person {int(person["ID"])}')
    plt.legend()
    plt.grid(True)
    plt.show()

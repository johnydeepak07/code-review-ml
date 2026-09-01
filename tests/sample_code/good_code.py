GRAVITY = 9.80665
PI = 3.14159


def calculate_orbital_velocity(radius: float, mass: float) -> float:
    """
    Calculate the orbital velocity given radius and central mass.

    Args:
        radius: Orbital radius in meters
        mass: Central body mass in kilograms

    Returns:
        Orbital velocity in meters per second
    """
    gravitational_constant = 6.674e-11
    return (gravitational_constant * mass / radius) ** 0.5
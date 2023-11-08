import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Boid parameters
n_boids = 120
width, height = 300, 300

# Initialize boid positions and velocities
positions = np.random.rand(n_boids, 2) * np.array([width, height])
velocities = np.random.randn(n_boids, 2)

def update_planets(planet_positions, planet_velocities, width, height, planet_economy, decay_rate=0.01, growth_rate=0.1, redistribution_rate=0.05):
    planet_positions += planet_velocities
    planet_positions %= np.array([width, height])
    
    # Add some randomness to the economy values
    planet_economy += np.random.normal(0, 0.1, size=n_planets)

    # Update economy values
    planet_economy *= (1 - decay_rate)
    planet_economy += growth_rate

    # Add some more randomness to the redistribution rate
    redistribution_rate += np.random.normal(0, 0.01)

    # Redistribute economy between planets with some chaos
    sorted_indices = np.argsort(planet_economy)
    redistribution_amount = redistribution_rate * planet_economy[sorted_indices[-1]]
    planet_economy[sorted_indices[-1]] -= redistribution_amount
    planet_economy[sorted_indices[0]] += redistribution_amount * np.random.rand() * 2

def count_boids_near_planets(positions, planets, radius=10):
    planet_distance = np.linalg.norm(positions[:, np.newaxis] - planets, axis=2)
    is_near = planet_distance < radius
    # Add some chaos to the boid counts
    boid_counts = np.sum(is_near, axis=0) + np.random.randint(-5, 5, size=len(planets))
    return boid_counts

# Define rules for boid behavior with some crazy modifications
def update_boids(positions, velocities, planets, planet_economy, combat_threshold=16, velocity_limit=None):
    # Compute distance between all pairs of boids and boids and planets with some randomness
    boid_distance = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2) * np.random.normal(1, 0.2, size=(n_boids, n_boids))
    planet_distance = np.linalg.norm(positions[:, np.newaxis] - planets, axis=2) * np.random.normal(1, 0.1, size=(n_boids, n_planets))

    # Find nearest planet for each boid with some confusion
    nearest_planet = np.argmin(planet_distance * np.random.rand(n_boids, n_planets), axis=1)
    nearest_planet_positions = planets[nearest_planet]
    nearest_planet_economy = planet_economy[nearest_planet]

    # Add some chaos to the attraction and repulsion rules
    attraction = nearest_planet_positions - positions + np.random.normal(0, 0.1, size=(n_boids, 2))
    attraction *= (1 / (planet_distance.min(axis=1) ** 2))[:, np.newaxis] 
    attraction *= nearest_planet_economy[:, np.newaxis] 
    repulsion = positions - nearest_planet_positions
    repulsion_mask = (planet_distance.min(axis=1) < 20)[:, np.newaxis]
    repulsion *= repulsion_mask
    repulsion += np.random.normal(0, 0.05, size=(n_boids, 2))


    # Update boid behavior with previous rules, planet attraction, and repulsion
    velocities += (
        0.015 * repulsion_rule(boid_distance, min_distance=3)
        + 0.011 * alignment_rule(boid_distance, velocities, neighborhood_radius=35)
        + 0.028 * cohesion_rule(boid_distance, positions, neighborhood_radius=35)
        + 0.015 * attraction  # Decrease the planetary gravity
        - 0.019 * repulsion  # Add repulsion effect when boids are too close to the planets
    )

    # Combat state
    combat_radius = 4
    combat_mask = (boid_distance > 0) & (boid_distance < combat_radius)
    combat_neighbor_counts = np.sum(combat_mask, axis=1)
    in_combat = combat_neighbor_counts >= combat_threshold

    if in_combat.any:
        combat_threshold += 1

    repulsion = repulsion_rule(boid_distance, min_distance=3)
    repulsion[in_combat] *= 1.3
    velocities[in_combat] += 0.25 * repulsion[in_combat]
    velocities[in_combat] += 0.12 * alignment_rule(boid_distance[in_combat], velocities[in_combat], neighborhood_radius=15)
    velocities[in_combat] -= 0.12 * cohesion_rule(boid_distance[in_combat], positions[in_combat], neighborhood_radius=15)
    velocities[in_combat] += velocities[in_combat]

    # Initialize ship health to 1
    ship_health = np.ones(n_boids)

    # Check if boids are stopping at planets
    boid_counts = count_boids_near_planets(positions, planet_positions)
    for i, count in enumerate(boid_counts):
        if count > 0:
            # Increase ship health if stopping at planet
            ship_health[i] += 0.1 * count

    # Check for combat
    combat_radius = 7
    combat_mask = (boid_distance > 0) & (boid_distance < combat_radius)
    combat_neighbor_counts = np.sum(combat_mask, axis=1)
    in_combat = combat_neighbor_counts >= combat_threshold

    if in_combat is not None:
        # Check for winning or losing combat
        winner_mask = np.zeros(n_boids, dtype=bool)
        loser_mask = np.zeros(n_boids, dtype=bool)
        for i in np.where(in_combat & in_combat)[0]:
            j = np.argmax(combat_mask[i])
            if i == j:
                continue
            if in_combat[j]:
                continue  # Skip if both are combative
            if np.random.rand() < 0.5:
                winner_mask[i] = True
                loser_mask[j] = True
                ship_health[i] += 0.2
                ship_health[j] -= 0.2
            else:
                winner_mask[j] = True
                loser_mask[i] = True
                ship_health[j] += 0.2
                ship_health[i] -= 0.2

        # Remove losing ships from the simulation
        positions[loser_mask] = np.nan
        velocities[loser_mask] = np.nan

    # Remove dead ships from the simulation
    positions[np.isnan(ship_health)] = np.nan
    velocities[np.isnan(ship_health)] = np.nan

    # Normalize velocities
    velocities /= np.linalg.norm(velocities, axis=1)[:, np.newaxis]

    # Optionally apply a velocity limit
    if velocity_limit is not None:
        velocities = np.clip(velocities, -velocity_limit, velocity_limit)

    # Update ship positions and health
    positions += velocities
    ship_health -= 0.01  # Decrease ship health over time

    # Kill ships with health less than or equal to 0
    positions[ship_health <= 0] = np.nan
    velocities[ship_health <= 0] = np.nan
    ship_health[ship_health <= 0] = np.nan

    # Periodic boundary conditions
    positions %= np.array([width, height])

    # Set color for combative and non-combative boids
    colors = ['red' if c else 'blue' for c in in_combat]

    # Update scatter plot with new colors
    sc.set_offsets(positions)
    sc.set_color(colors)

    # Normalize velocities
    velocities /= np.linalg.norm(velocities, axis=1)[:, np.newaxis]

    # Optionally apply a velocity limit
    if velocity_limit is not None:
        velocities = np.clip(velocities, -velocity_limit, velocity_limit)

    positions += velocities

    # Periodic boundary conditions
    positions %= np.array([width, height])

    boid_counts = count_boids_near_planets(positions, planets)
    
    # Boids visiting a planet increase the planet's economy
    for i, count in enumerate(boid_counts):
        planet_economy[i] += count * 0.1

def repulsion_rule(distance, min_distance):
    mask = (distance > 0) & (distance < min_distance)
    return np.sum((positions[:, np.newaxis] - positions) * mask[:,:,np.newaxis], axis=1)

def alignment_rule(distance, velocities, neighborhood_radius):
    mask = (distance > 0) & (distance < neighborhood_radius)
    return np.sum(velocities[:, np.newaxis] * mask[:,:,np.newaxis], axis=1) / (np.sum(mask, axis=1)[:, np.newaxis] + 1e-6) - velocities

def cohesion_rule(distance, positions, neighborhood_radius):
    mask = (distance > 0) & (distance < neighborhood_radius)
    return np.sum(positions[:, np.newaxis] * mask[:,:,np.newaxis], axis=1) / (np.sum(mask, axis=1)[:, np.newaxis] + 1e-6) - positions

n_planets = 5
planet_positions = np.random.rand(n_planets, 2) * np.array([width, height])

planet_velocities = np.random.rand(n_planets, 2) * 0.2 - 0.1  # Initialize random velocities for planets

planet_economy = np.random.rand(n_planets) * 20 + 10  # Initialize random economic values for planets

fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(positions[:, 0], positions[:, 1])
ax.set_xlim(0, width)
ax.set_ylim(0, height)

economy_text = ax.text(0.01, 0.99, "", transform=ax.transAxes, verticalalignment="top")

# Add planets to the plot
planet_sc = ax.scatter(planet_positions[:, 0], planet_positions[:, 1], s=90, c='orange', marker='o')

def update(frame):
    update_boids(positions, velocities, planet_positions, planet_economy, 4, 3)
    update_planets(planet_positions, planet_velocities, width, height, planet_economy)

    sc.set_offsets(positions)
    planet_sc.set_offsets(planet_positions)  # Update planet positions on the plot

    boid_counts = count_boids_near_planets(positions, planet_positions)
    economy_text.set_text(
        "\n".join(
            f"Planet {i}: Economy={economy:.1f}, Boids={count}"
            for i, (economy, count) in enumerate(zip(planet_economy, boid_counts))
        )
    )

    return sc, planet_sc, economy_text

ani = FuncAnimation(fig, update, frames=range(500), interval=50, repeat=True, blit=True)
plt.show()

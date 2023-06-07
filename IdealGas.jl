#========================================================================================#
"""
	IdealGas
änderung
Extending SimpleParticles to conserve kinetic energy and momentum.

Author: Niall Palfreyman, 20/01/23
"""
module IdealGas


include("AgentTools.jl")
using Agents, LinearAlgebra, GLMakie, InteractiveDynamics, .AgentTools

#-----------------------------------------------------------------------------------------
# Module types:
#-----------------------------------------------------------------------------------------
"""
	Particle

The populating agents in the IdealGas model.
"""
@agent Particle ContinuousAgent{2} begin
	mass::Float64					# Particle's mass
	speed::Float64					# Particle's speed
	radius::Float64					# Particle's radius
	prev_partner::Int				# Previous collision partner id
end

"Standard value that is definitely NOT a valid agent ID"
const non_id = -1

#-----------------------------------------------------------------------------------------
# Module methods:
#-----------------------------------------------------------------------------------------
"""
	idealgas( kwargs)

Create and initialise the IdealGas model.
"""
function idealgas(;
    n_particles = 50,				# Number of Particles in box
	mass_u = 4.0,					# Helium Gas mass in atomic mass units
	init_temp = 300.0,				# Initial temperature of the gas in Kelvin
	radius = 1,						# Radius of Particles in the box
    extent = (100, 50),				# Extent of Particles space

)
    space = ContinuousSpace(extent; spacing = radius/1.5)

	properties = Dict(
		:n_particles	=> n_particles,
		:init_temp		=> init_temp,
	)

    box = ABM( Particle, space; properties, scheduler = Schedulers.Randomly())

    k = 1.38e-23  # Boltzmann constant in J/K
	mass_kg = mass_u * 1.66053906660e-27  # Convert atomic/molecular mass to kg
	max_speed = 1000.0  # Maximum speed in m/s
	for _ in 1:n_particles
		vel = Tuple( 2rand(2).-1)
		vel = vel ./ norm(vel)  # ALWAYS maintain normalised state of vel!
		speed = sqrt((3 * k * box.properties[:init_temp]) / mass_kg)  # Initial speed based on temperature
		speed = scale_speed(speed, max_speed)  # Scale speed to avoid excessive velocities
        add_agent!( box, vel, mass_kg, speed, radius, non_id)
	end

    return box
end

#-----------------------------------------------------------------------------------------
"""
	agent_step!( me, box)

This is the heart of the IdealGas model: It calculates how Particles collide with each other,
while conserving momentum and kinetic energy.
"""
function agent_step!(me::Particle, box::ABM)
	her = random_nearby_agent( me, box, 2*me.radius)	# Grab nearby particle
	if her === nothing
		# No new partners - forget previous collision partner:
		me.prev_partner = non_id
	elseif her.id < me.id && her.id != me.prev_partner
		# New collision partner has not already been handled and is not my previous partner:
		me.prev_partner = her.id							# Update previous partners to avoid
		her.prev_partner = me.id							# repetitive juddering collisions.
		cntct = (x->[cos(x),sin(x)])(2rand()pi)				# Unit vector to contact point with partner
		Rctct = [cntct[1] cntct[2]; -cntct[2] cntct[1]]		# Rotation into contact directn coords
		Rback = [cntct[1] -cntct[2]; cntct[2] cntct[1]]		# Inverse rotation back to world coords

		# Rotate velocities into coordinates directed ALONG and PERPendicular to contact direction:
		myAlongVel, myPerpVel = me.speed * Rctct * collect(me.vel)					# My velocity
		herAlongVel, herPerpVel = her.speed * Rctct * collect(her.vel)				# Her velocity
		cmAlongVel = (me.mass*myAlongVel + her.mass*herAlongVel)/(me.mass+her.mass)	# C of M velocity

		# Calculate collision effects along contact direction (perp direction is unaffected):
		myAlongVel = 2cmAlongVel - myAlongVel
		herAlongVel = 2cmAlongVel - herAlongVel

		# Rotate collision effects on both me and her back into world coordinates:
		me.speed = hypot(myAlongVel,myPerpVel)
		if me.speed != 0.0
			me.vel = Tuple(Rback*[myAlongVel,myPerpVel])
			me.vel = me.vel ./ norm(me.vel)
		end
		her.speed = hypot(herAlongVel,herPerpVel)
		if her.speed != 0.0
			her.vel = Tuple(Rback*[herAlongVel,herPerpVel])
			her.vel = her.vel ./ norm(her.vel)
		end
	end
	move_agent!(me, box, me.speed)
end

#-----------------------------------------------------------------------------------------
"""
	momentum( particle)

Return the momentum of this particle.
"""
function momentum(particle)
	particle.mass * particle.speed * collect(particle.vel)
end

#-----------------------------------------------------------------------------------------
"""
	kinetic_energy( particle)

Return the kinetic energy of this particle.
"""
function kinetic_energy(particle)
	particle.mass * particle.speed^2 / 2
end

#-----------------------------------------------------------------------------------------
"""
	scale_speed(speed, max_speed)

Scales a speed value to the interval [0,1] based on the provided max_speed.
"""
function scale_speed(speed, max_speed)
	if speed > max_speed
		speed = max_speed
	end
    return speed / max_speed
end

#-----------------------------------------------------------------------------------------

"""
	calc_temperature

Return the temperature of the system.
"""

function calc_temperature(box::ABM)
    total_energy = 0.0
    for p in allagents(box)
        total_energy += kinetic_energy(p)
    end
    # k = 1.38e-23 is the Boltzmann constant
    T = total_energy / (1.38e-23 * 3/2 * box.properties[:n_particles])
    box.properties[:init_temp] = T
    return T
end



#-----------------------------------------------------------------------------------------
"""
	model_step!( model)

This is the heart of the IdealGas model: It calculates how Particles collide with each other
while conserving momentum and kinetic energy.
"""
function model_step!(model)
    println("Temperature: ", calc_temperature(model))
    println("Pressure: ", calc_pressure(model))
end


#------------------------------------------------------------------------------------------

"""
	calc_pressure(box)

Return the pressure of the system.
"""
function calc_pressure(box::ABM)
    R = 8.314 # Gaskonstante in J/(mol·K)
    n = box.properties[:n_particles] # Anzahl der Moleküle (angenommen, jedes Partikel repräsentiert ein Molekül)
    V = box.space.extent[1] * box.space.extent[2] # Volumen der Box, unter der Annahme, dass sie 2D ist
    T = calc_temperature(box) # Durchschnittstemperatur

    P = n * R * T / V
    return P
end

#----------------------------------------------------------------------------------------

"""
	demo()

Run a simulation of the IdealGas model.
"""

params = Dict(
		:n_particles => 20:1:100,
		:init_temp => 100.0:1.0:1000.0
	)

function demo()
	box = idealgas()
	
	playground, = abmplayground( box, idealgas;
	agent_step!,
	model_step!,
	params
)

	playground 
end

end	# of module IdealGas
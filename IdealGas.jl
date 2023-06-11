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
	volume = [10.0, 2.0, 0.01], 						#Reale Maße des Containers
	temp = 273.15,										# Initial temperature of the gas in Kelvin
	temp_old = temp,									# Initial temperature of the gas in Kelvin
	pressure_bar = 1.0,									# Initial pressure of the gas in bar
	pressure_bar_old = pressure_bar,								# Initial pressure of the gas in bar
	pressure_pa =  pressure_bar*1e5,									# Initial pressure of the gas in Pascal
	n_mol = pressure_pa * volume[1] * volume[2] * volume[3] / (8.314*temp),				# Number of mol
	init_n_mol = n_mol,
	real_n_particles = n_mol * 6.022e23,							# Real number of Particles in box
    n_particles = real_n_particles/1e23,				# Number of Particles in simulation box
	mass_u = 4.0,										# Helium Gas mass in atomic mass units
	radius = 1,											# Radius of Particles in the box
	e_inner = 3/2 * real_n_particles * temp * 8.314,	# Inner energy of the gas
	extent = (volume[2]*100, volume[1]*100)			# Extent of Particles space
)
    space = ContinuousSpace(extent; spacing = radius/1.5)

	properties = Dict(
		:n_particles	=> n_particles,
		:temp		=> temp,
		:e_inner	=> e_inner,
		:pressure_pa	=> pressure_pa,
		:pressure_bar	=> pressure_bar,
		:real_n_particles	=> real_n_particles,
		:n_mol		=> n_mol,
		:volume	=> volume,
		:temp_old	=> temp_old,
		:pressure_bar_old	=> pressure_bar_old,
		:init_n_mol	=> init_n_mol,
	)

    box = ABM( Particle, space; properties, scheduler = Schedulers.Randomly())

    k = 1.38e-23  # Boltzmann constant in J/K
	mass_kg = mass_u * 1.66053906660e-27  # Convert atomic/molecular mass to kg
	max_speed = 1000.0  # Maximum speed in m/s
	for _ in 1:n_particles
		vel = Tuple( 2rand(2).-1)
		vel = vel ./ norm(vel)  # ALWAYS maintain normalised state of vel!
		speed = sqrt((3 * k * box.temp) / mass_kg)  # Initial speed based on temperature
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
function calc_temperature(model::ABM)   
    # T = Ekin / (k * 2/3 * N ); Boltzmann constant k = 1.38e-23
	P = model.pressure_pa
	R = 8.314 # Gaskonstante in J/(mol·K)
	n = model.n_mol # Anzahl der Moleküle
	V = model.volume[1] * model.volume[2] * model.volume[3]
	T = (P*V)/(R*n)
	return T
end



#-----------------------------------------------------------------------------------------
"""
	model_step!( model)


"""
function model_step!(model::ABM)
	println("T = ", model.temp, " K")
	println("T temp_old = ", model.temp_old, " K")
	println("P = ", model.pressure_pa, " Pa")
	println("P pressure_bar_old = ", model.pressure_bar_old, " Pa")
	println("n_mol = ", model.n_mol)
	println("real_n_particles = ", model.real_n_particles)
	println("n_particles = ", model.n_particles)
	print("\n")
	if model.pressure_bar_old != model.pressure_bar
		model.n_mol = model.init_n_mol * model.pressure_bar
		model.real_n_particles = calc_real_n_particles(model)
		model.n_particles = model.real_n_particles / 1e23
		model.pressure_pa = model.pressure_bar * 1e5
		model.temp = calc_temperature(model)
	end
	if model.temp != model.temp_old
		model.pressure_pa = calc_pressure(model)
		model.pressure_bar = model.pressure_pa / 1e5
	end
	model.pressure_bar_old = model.pressure_bar
	model.temp_old = model.temp
    model.e_inner = 3/2 * model.real_n_particles * model.temp * 8.314
end


#------------------------------------------------------------------------------------------
"""
	calc_n_mol(model)

Return the number of molecules in the system.
"""
function calc_n_mol(model::ABM)
	return model.pressure_bar * 1e5 * model.volume[1] * model.volume[2] * model.volume[3] / (8.314*model.temp)
end

#------------------------------------------------------------------------------------------
"""
	calc_real_n_particles(model)

Return the number of particles in the system.
"""
function calc_real_n_particles(model::ABM)
	return model.n_mol * 6.022e23
end

#------------------------------------------------------------------------------------------
"""
	calc_pressure(box)

Return the pressure of the system.
"""
function calc_pressure(model::ABM)
    R = 8.314 # Gaskonstante in J/(mol·K)
    n = model.n_mol # Anzahl der Moleküle (angenommen, jedes Partikel repräsentiert ein Molekül)
    V = model.volume[1] * model.volume[2] * model.volume[3] # Volumen des Behälters
    T = model.temp # Durchschnittstemperatur der Moleküle
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
		:temp => 100.0:1.0:1000.0,
		:pressure_bar=> 0.0:0.1:12.0
	)

function demo()
	box = idealgas()

	inner_energy(box) = box.e_inner
	temperature(box) = box.temp
	pressure_bar(box) = box.pressure_bar
	mdata = [inner_energy, temperature, pressure_bar]
	
	playground, = abmplayground( box, idealgas;
	agent_step!,
	model_step!,
	mdata,
	params
)

	playground 
end

end	# of module IdealGas
#========================================================================================#
"""
	IdealGas
änderung
Extending SimpleParticles to conserve kinetic energy and momentum.

Author: Niall Palfreyman, 20/01/23
"""
module IdealGas


include("AgentTools.jl")
using Agents, LinearAlgebra, GLMakie, InteractiveDynamics, .AgentTools, GeometryBasics, Observables

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
	last_bounce::Float64            # 
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
	gases = Dict("Helium" => 4.0, "Hydrogen" => 1.0, "Oxygen" => 32.0),					# Gas types
	volumes = Dict("Gasflasche" => [10.0, 2.0, 0.01], "Gastank" => [30, 100, 0.01]),	# Volume of containers
	volume = [10.0, 2.0, 0.01], 														# Reale Maße des Containers
	temp = 273.15,																		# Initial temperature of the gas in Kelvin
	temp_old = copy(temp),																# Initial temperature of the gas in Kelvin
	pressure_bar = 1.0,																	# Initial pressure of the gas in bar
	pressure_bar_old = copy(pressure_bar),												# Initial pressure of the gas in bar
	pressure_pa =  pressure_bar*1e5,													# Initial pressure of the gas in Pascal
	n_mol = pressure_pa * volume[1] * volume[2] * volume[3] / (8.314*temp),				# Number of mol
	init_n_mol = copy(n_mol), 															# Initial number of mol
	real_n_particles = n_mol * 6.022e23,												# Real number of Particles in box
    n_particles = real_n_particles/1e23,												# Number of Particles in simulation box
	molare_masse = 4.0,																	# Helium Gas mass in atomic mass units
	mass_kg = molare_masse * 1.66053906660e-27,											# Convert atomic/molecular mass to kg
	mass_gas = round(n_mol * molare_masse, digits=3),									# Mass of gas
	radius = 8.0,																		# Radius of Particles in the box
	e_inner = 3/2 * real_n_particles * temp * 8.314,									# Inner energy of the gas
	entropy = 0.0,
	extent = (volume[2]*300.0, volume[1]*100.0),												# Extent of Particles space
)
    space = ContinuousSpace(extent; spacing = radius/2)

	properties = Dict(
		:n_particles	=> n_particles,
		:temp		=> temp,
		:e_inner	=> e_inner,
		:entropy 	=> entropy,
		:pressure_pa	=> pressure_pa,
		:pressure_bar	=> pressure_bar,
		:real_n_particles	=> real_n_particles,
		:n_mol		=> n_mol,
		:volume	=> volume,
		:temp_old	=> temp_old,
		:pressure_bar_old	=> pressure_bar_old,
		:init_n_mol	=> init_n_mol,
		:gases		=> gases,
		:molare_masse		=> molare_masse,
		:mass_kg		=> mass_kg,
		:volumes	=> volumes,
		:mass_gas	=> mass_gas,
		:step => 0,
		
		:placeholder => 0.0,
	)

    box = ABM( Particle, space; properties, scheduler = Schedulers.Randomly())

    k = 1.38e-23  									# Boltzmann constant in J/K
	max_speed = 1000.0  							# Maximum speed in m/s
	for _ in 1:n_particles
		vel = Tuple( 2rand(2).-1)
		vel = vel ./ norm(vel)  					# ALWAYS maintain normalised state of vel!
		speed = sqrt((3 * k * box.temp) / mass_kg)  # Initial speed based on temperature
		speed = scale_speed(speed, max_speed)  		# Scale speed to avoid excessive velocities
        add_agent!( box, vel, mass_kg, speed, radius, non_id, -Inf)
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
    her = random_nearby_agent(me, box, 2*me.radius)   # Grab nearby particle
    
    if her !== nothing && her.id < me.id && her.id != me.prev_partner
        # New collision partner has not already been handled and is not my previous partner:
        me.prev_partner = her.id           # Update previous partners to avoid repetitive juddering collisions.
        her.prev_partner = me.id           # ditto for the other agent.

        # Compute relative position and velocity:
        rel_pos = me.pos .- her.pos
        rel_vel = me.vel .- her.vel

        # Compute collision impact vector:
        distance_sq = sum(rel_pos .^ 2)
        velocity_dot = sum(rel_vel .* rel_pos)
        impulse = 2 * me.mass * her.mass / (me.mass + her.mass) * velocity_dot ./ distance_sq .* rel_pos

        # Update velocities according to the impulse
		me.vel = (me.vel[1] - (impulse ./ me.mass)[1], me.vel[2] - (impulse ./ me.mass)[2])
		her.vel = (her.vel[1] + (impulse ./ her.mass)[1], her.vel[2] + (impulse ./ her.mass)[2])

        # Update speeds based on new velocities
        me.speed = norm(me.vel)
        her.speed = norm(her.vel)
		
		#me = foo(me,box)
    end

	# Überprüfen, ob das Partikel nahe am Rand ist
	check_particle_near_border!(me, box)  # Aufruf der neuen Funktion

	#if me.pos[2] > 500
		#println("hello")
	#	me.pos = (me.pos[1], 20)
	#end

    move_agent!(me, box, me.speed)
end

#----------------------------------------------------------------------------------------

function check_particle_near_border!(me, box)
    x, y = me.pos

    if x < 1.8 && box.properties[:step] - me.last_bounce > 3
        me.vel = (-me.vel[1], me.vel[2])
        me.last_bounce = box.properties[:step]
    elseif x > box.space.extent[1] - 1.8 && box.properties[:step] - me.last_bounce > 3
        me.vel = (-me.vel[1], me.vel[2])
        me.last_bounce = box.properties[:step]
    end
    if y < 1.8 && box.properties[:step] - me.last_bounce > 3
        me.vel = (me.vel[1], -me.vel[2])
        me.last_bounce = box.properties[:step]			
    elseif y > box.space.extent[2] - 1.8 && box.properties[:step] - me.last_bounce > 3 
        me.vel = (me.vel[1], -me.vel[2])
        me.last_bounce = box.properties[:step]
    end

	# Überprüfen, ob y > 500 und falls ja, setzen Sie y auf 500 und invertieren Sie die y-Geschwindigkeit
    if y > 500
        me.pos = (x, 498)
        me.vel = (me.vel[1], -me.vel[2])
    end
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
	"""
	println("T = ", model.temp, " K")
	println("P = ", model.pressure_pa, " Pa")
	println("real_n_particles = ", model.real_n_particles)
	println("n_particles = ", model.n_particles)
	print("\n")
	println(model.molare_masse)
	println("volume: ", model.volume)
	println("molare_masse: ", model.molare_masse, " g/mol")
	println("mass_kg: ", model.mass_kg, " kg")
	"""

	pressure_pa = model.n_mol * 8.314 * model.temp / (model.volume[1] * model.volume[2] * model.volume[3])
	model.pressure_pa = round(pressure_pa, digits=3)
	model.pressure_bar = round(model.pressure_pa / 1e5, digits=3)

	"""
	println("pressure_pa: " , model.pressure_pa, " Pa")
	println("pressure_bar: ", model.pressure_bar, " Bar")
	println("n_mol: ", model.n_mol, " mol")
	println("mass_gas: ", model.mass_gas, " g")
	println("\n")
	println("\n")
	"""
	model.entropy = 0.0

	model.properties[:step] += 1 # steps werden gezählt, notwendig für die Abstoßung am Rand

    #model.e_inner = 3/2 * model.real_n_particles * model.temp * 8.314

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
		:temp => 100.0:1.0:1000.0,
		:placeholder => 0:1:10,
	)

	function demo()
		box = idealgas()
	
		entropy(box) = box.entropy
		mdata = [entropy]
		mlabels = ["Entropie(Platzhalter)"]
	
		playground,abmobs = abmplayground( box, idealgas;
			agent_step!,
			model_step!,
			mdata,
			mlabels,
			params,
			figure = (; resolution = (1300, 750)),
			ac = :skyblue3,
			as = 20.0
		)
		# Figure Objekten neues Layout zuweisen durch feste Reihenfolge in figure.content[i]
		model_plot = playground.content[1]	# Box 	
		playground[0:2,0] = model_plot
		entropy_plot = playground.content[9]
		playground[0:1,2][1,0:1] = entropy_plot
		# Sliders
		playground[2,1] = playground.content[2]
		playground[2,2] = playground.content[7]
		# Buttons
		gl_buttons = playground[3,1] = GridLayout()
		gl_buttons[0,2] = playground.content[3]
		gl_buttons[0,3] = playground.content[4]
		gl_buttons[0,4] = playground.content[5]
		gl_buttons[0,5] = playground.content[6]
		playground[3,1][0,8] = playground.content[8]	# Update Button	

		# grid_layout = playground[3,1] = GridLayout()
		# count_layout = grid_layout[1,1] = GridLayout()
		gl_dropdowns = playground[3,0] = GridLayout()
		gl_labels = playground[0,1] = GridLayout()
		# gas_dropdown = Menu(count_layout[1,1], options = keys(box.gases), default = "Helium")
		gas_dropdown = Menu(gl_dropdowns[0,0], options = keys(box.gases), default = "Helium")
		# volume_dropdown = Menu(count_layout[2,1], options = keys(box.volumes), default = "Gasflasche")
		volume_dropdown = Menu(gl_dropdowns[0,1], options = keys(box.volumes), default = "Gasflasche")
		#volume_slider = SliderGrid(playground[1,1], (label = "Höhe: ", range = 0/50:0.1:10.0, startvalue=10.0))
		#playground[5,1] = volume_slider
		pressure_label = Label(gl_labels[2,0], "Druck: " * string(box.pressure_bar)* " Bar", fontsize=22)
		mass_label = Label(gl_labels[3,0], "Masse: " * string(box.mass_gas)* " g", fontsize=22)
		volume_label = Label(gl_labels[1,0], "Volumen: " * string(box.volume[1] * box.volume[2] * box.volume[3])* " m³ ; " * string(box.volume[1] * box.volume[2] * box.volume[3] * 1000) * " L", fontsize=22)
		# Platzhalter Label
		Label(gl_labels[4,0], "Platzhalter: 0.0 ", fontsize=22)
		Label(gl_labels[5,0], "Platzhalter: 0.0 ", fontsize=22)

		on(abmobs.model) do _

			pressure_label.text[] = string("Druck: ", string(box.pressure_bar), " Bar")
			if box.mass_gas > 999.9
				mass_label.text[] = string("Masse: ", string(round(box.mass_gas/1000, digits=3), " kg"))
			else
				mass_label.text[] = string("Masse: ", string(box.mass_gas), " g")
			end
			volume_label.text[] = string("Volumen: ", string(box.volume[1] * box.volume[2] * box.volume[3]), " m³ ; " * string(box.volume[1] * box.volume[2] * box.volume[3] * 1000) * " L")
		end

		on(gas_dropdown.selection) do selected_gas
			new_molare_masse = box.gases[selected_gas]
			box.molare_masse = new_molare_masse
			box.mass_kg = new_molare_masse * 1.66054e-27
			box.mass_gas = round(box.n_mol * box.molare_masse, digits=3)
			
		end

		on(volume_dropdown.selection) do selected_volume
			new_volume = box.volumes[selected_volume]
			box.volume = new_volume
			box.n_mol = box.pressure_pa * box.volume[1] * box.volume[2] * box.volume[3] / (8.314*box.temp)
			box.mass_gas = round(box.n_mol * box.molare_masse, digits=3)
		
			
		end
		playground
	end

end	# of module IdealGas
#========================================================================================#
"""
	IdealGas

Extending SimpleParticles to conserve kinetic energy and momentum.

Author: Francisco Hella, Felix Rollbühler, Melanie Heinrich, Jan Wichmann, 22/06/23
"""
module IdealGas


include("AgentTools.jl")
include("TD_Physics.jl")
using Agents, LinearAlgebra, GLMakie, InteractiveDynamics, .AgentTools, .TD_Physics

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
	last_bounce::Float64			
end

"Standard value that is definitely NOT a valid agent ID"
const non_id = -1
const R = 8.314 # Gaskonstante in J/(mol·K)

#-----------------------------------------------------------------------------------------
# Module methods:
#-----------------------------------------------------------------------------------------
"""
	idealgas( kwargs)

Create and initialise the IdealGas model.
"""
function idealgas(;
	gases = Dict("Helium" => 4.0, "Hydrogen" => 1.0, "Oxygen" => 32.0),					# Gas types
	modes = Dict("Temperatur : Druck" => "temp-druck",									# Modes of operation
				"Temperatur : Volumen" => "temp-vol",
				"Druck : Temperatur" => "druck-temp",
				"Druck : Volumen" => "druck-vol",
				"Volumen : Temperatur" => "vol-temp",
				"Volumen : Druck" => "vol-druck",										
				"Mol : Temperatur" => "mol-temp",
				"Mol : Druck" => "mol-druck"
				),
	mode = "temp-druck",																		# actual Mode of operation
	total_volume 			= 0.2,																# Initial volume of the container
	volume 					= calc_total_vol_dimension(total_volume), 							# Dimensions of the container
	topBorder 				= total_volume/5.0,													# Top border of the container
	temp 					= 293.15,															# Initial temperature of the gas in Kelvin
	temp_old 				= 293.15,															# Old temperature of the gas in Kelvin
	pressure_bar 			= 1.0,																# Initial pressure of the gas in bar
	pressure_pa 			= pressure_bar*1e5,													# Initial pressure of the gas in Pascal
	n_mol 					= pressure_pa * total_volume / (8.314*temp),						# Number of mol
	init_n_mol 				= copy(n_mol), 														# Initial number of mol
	real_n_particles 		= n_mol * 6.022e23,													# Real number of Particles in model: Reduction for simplicity
    n_particles 			= real_n_particles/1e22/8,											# Number of Particles in simulation model
	n_particles_old 		= copy(n_particles),												# Old number of Particles in simulation model
	molar_mass 				= 4.0,																# Helium Gas mass in atomic mass units
	mass_kg 				= molar_mass * 1.66053906660e-27,									# Convert atomic/molecular mass to kg
	mass_gas 				= round(n_mol * molar_mass, digits=3),								# Mass of gas
	radius 					= 8.0,																# Radius of Particles in the model
	e_internal 				= 3/2 * n_mol * 8.314 * temp,										# Internal energy of the gas
	entropy_change 			= 0.0,																# Change in entropy of the gas
	step 					= 0,																# Step counter
	max_speed 				= 8000.0,  															# Maximum speed of the particles in m/s
	extent 					= (500,500),														# Extent of Particles space
)
    space = ContinuousSpace(extent; spacing = 2.5)												# Create the space for the particles

	properties = Dict(																			
		:n_particles		=> n_particles,
		:temp				=> temp,
		:temp_old			=> temp_old,
		:total_volume		=> total_volume,
		:e_internal			=> e_internal,
		:entropy_change 	=> entropy_change,
		:pressure_pa		=> pressure_pa,
		:pressure_bar		=> pressure_bar,
		:real_n_particles	=> real_n_particles,
		:n_mol				=> n_mol,
		:n_particles_old	=> n_particles_old,
		:volume				=> volume,
		:init_n_mol			=> init_n_mol,
		:gases				=> gases,
		:molar_mass		=> molar_mass,
		:mass_kg			=> mass_kg,
		:mass_gas			=> mass_gas,
		:topBorder 			=> topBorder,
		:step 				=> step,
		:modes				=> modes,
		:mode				=> mode,
		:radius				=> radius,
		:max_speed			=> max_speed,
	)


    model = ABM( Particle, space; properties, scheduler = Schedulers.Randomly())

	for _ in 1:n_particles
		vel = Tuple( 2rand(2).-1)
		vel = vel ./ norm(vel)  # ALWAYS maintain normalised state of vel!
		speed = sqrt((3 * R * model.temp) / (model.molar_mass / 1000))  # Initial speed based on temperature
		speed = scale_speed(speed, max_speed)  				# Scale speed to avoid excessive velocities
        add_agent!( model, vel, mass_kg, speed, radius, non_id, -Inf)
	end

    return model
end
#TODO: Zur Volumenveränderung zwei Buttons, erhöhen und erniedrigen
#TODO: Volumenveränderung beschleunigt die Teilchen die gegen die Seite von der Arbeitverrichtet wird
#-----------------------------------------------------------------------------------------
"""
calc_total_vol_dimension( me, model)

Calculates volume/dimension of a 3D-Space with [x, y=5, z=1], based on a given value of total volume.
"""
function calc_total_vol_dimension(volume, x_axis_vol=5.0)
 	y_axis_vol = volume/x_axis_vol
 	return [y_axis_vol, x_axis_vol, 1.0] 
end
#-----------------------------------------------------------------------------------------
"""
	agent_step!( me, model)

This is the heart of the IdealGas model: It calculates how Particles collide with each other,
while conserving momentum and kinetic energy.
"""
function agent_step!(me::Particle, model::ABM)
    her = random_nearby_agent(me, model, 2*me.radius)   # Grab nearby particle
    
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
    end

	check_particle_near_border!(me, model)
		
	move_agent!(me, model, me.speed)
end
#----------------------------------------------------------------------------------------

function check_particle_near_border!(me, model)
    x, y = me.pos

    if x < 1.8 && model.step - me.last_bounce > 3
        me.vel = (-me.vel[1], me.vel[2])
        me.last_bounce = model.step
    elseif x > model.space.extent[1] - 1.8 && model.step - me.last_bounce > 3
        me.vel = (-me.vel[1], me.vel[2])
        me.last_bounce = model.step
    end
    if y < 1.8 && model.step - me.last_bounce > 3
        me.vel = (me.vel[1], -me.vel[2])
        me.last_bounce = model.step			
    elseif y > model.space.extent[2] - 1.8 && model.step - me.last_bounce > 3 
        me.vel = (me.vel[1], -me.vel[2])
        me.last_bounce = model.step
    end

	# Überprüfen, ob y > 500 und falls ja, setzen Sie y auf 500 und invertieren Sie die y-Geschwindigkeit
    # if y > 500
    #     me.pos = (x, 498)
    #     me.vel = (me.vel[1], -me.vel[2])
    # end
end
#-----------------------------------------------------------------------------------------

"""
	model_step!( model)

	calculate the quantities, based on the chosen mode (Specifies which variables are constant)
"""
function model_step!(model::ABM)

	scale_agent_speed(model) # Scale speed to avoid excessive velocities

	model.entropy_change = calc_entropy_change(model)
	
	model.e_internal = calc_internal_energy(model)

	model.step += 1.0


end

#----------------------------------------------------------------------------------------

"""
	demo()

Run a simulation of the IdealGas model.
"""



	function demo()
		model = idealgas()
		#params = Dict(:temp => 100.0:1.0:1000.0,:total_volume => 0:0.1:30,:placeholder => 0:1:10)
	
		entropy(model) = model.entropy_change
		mdata = [entropy]
		mlabels = ["ΔS in [J/K] (Entropieänderung)"]

		# Resolution of Makie Window based on monitor size 
		monitor = GLMakie.GLFW.GetPrimaryMonitor()
		width = GLMakie.MonitorProperties(monitor).videomode.width
		height = GLMakie.MonitorProperties(monitor).videomode.height
	
		playground,abmobs = abmplayground( model, idealgas;
			agent_step!,
			model_step!,
			mdata,
			mlabels,
			#params,
			figure = (; resolution = (width*0.75, height*0.75)),
			ac = :skyblue3,
			as = 30.0
		)

		# Figure Objekten neues Layout zuweisen durch feste Reihenfolge in figure.content[i]
		model_plot = playground.content[1]	# model 	
		playground[0:2,0] = model_plot
		entropy_plot = playground.content[7]
		playground[0:1,2][1,0:1] = entropy_plot

		# Sliders
		playground[2,1] = playground.content[2]
		slider_space = playground[2,2] = GridLayout()

		# Buttons to change volume 
		vol_change_btns = playground[1,1] = GridLayout()

		# Buttons
		gl_buttons = playground[3,1] = GridLayout()
		gl_buttons[0,2] = playground.content[3]
		gl_buttons[0,3] = playground.content[4]
		gl_buttons[0,4] = playground.content[5]
		gl_buttons[0,5] = playground.content[6]

		# GridLayouts
		gl_sliders = playground[4,:] = GridLayout()
		gl_dropdowns = playground[3,0] = GridLayout()
		gl_labels = playground[0,1] = GridLayout()

		# Dropdowns
		gas_dropdown = Menu(gl_dropdowns[0,0], options = keys(model.gases), default = "Helium")
		mode_dropdown = Menu(gl_dropdowns[0,1], options = keys(model.modes), default = "Temperatur : Druck")

		# Labels
		#pressure_label = Label(gl_labels[2,0], "Druck: " * string(round(model.pressure_bar, digits=2))* " Bar", fontsize=22)
		mass_label = Label(gl_labels[0,0], "Masse: " * string(model.mass_gas)* " g", fontsize=22)
		#volume_label = Label(gl_labels[1,0], "Volumen: " * string(round(model.total_volume, digits=2))* " m³ ; " * string(round(model.total_volume * 1000, digits=2)) * " L", fontsize=22)
		e_internal_label = Label(gl_labels[1,0], "Eᵢ: " * string(round(model.e_internal, digits=2)) * " J", fontsize=22)
		
		#Custom Buttons
		increase_vol_btn = Button(vol_change_btns[0,1:2], label = "Increase\nVolumen")# = print("increase"))#increase_vol_const())
		pause_vol_btn = Button(vol_change_btns[0,3], label = "Pause")
		decrease_vol_btn = Button(vol_change_btns[0,4:5], label = "Decrease\nVolumen")# = print("decrease"))#decrease_vol_const())

	
		#TODO: Hier volumen change funktionen aufrufen
		on(increase_vol_btn.clicks) do _
			println("increase_vol_btn")
		end  

		on(pause_vol_btn.clicks) do _
			println("pause volume change")
		end 

		on(decrease_vol_btn.clicks) do _
			println("decrease_vol_btn")
		end

		# multipliers and quotients to set min, max and step of sliders based on current values
		slider_multiplier_min = 0.5
		slider_multiplier_max = 1.5
		slider_step_quotient = 100.0

		temp_slider_label = Label(slider_space[0,0], "Temperatur: ", fontsize=16)
		temp_slider = Slider(slider_space[0,1], range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max, startvalue=293.15)
		temp_slider_value = Label(slider_space[0,2], string(temp_slider.value[]) * " K")


		pressure_slider_bar_label = Label(slider_space[1,0], "Druck: ", fontsize=16)
		pressure_slider_bar = Slider(slider_space[1,1], range = model.pressure_bar*slider_multiplier_min:model.pressure_bar/slider_step_quotient:model.pressure_bar*slider_multiplier_max, startvalue=1.0)
		pressure_slider_bar_value = Label(slider_space[1,2], string(pressure_slider_bar.value[]) * " Bar")

		pressure_slider_pa_label = Label(slider_space[2,0], "Druck: ", fontsize=16)
		pressure_slider_pa = Slider(slider_space[2,1], range = model.pressure_pa*slider_multiplier_min:model.pressure_pa/slider_step_quotient:model.pressure_pa*slider_multiplier_max, startvalue=model.pressure_pa)
		pressure_slider_pa_value = Label(slider_space[2,2], string(pressure_slider_pa.value[]) * " Pa")

		volume_slider_label = Label(slider_space[3,0], "Volumen: ", fontsize=16)
		volume_slider = Slider(slider_space[3,1], range = float(model.total_volume*slider_multiplier_min):model.total_volume/slider_step_quotient:float(model.total_volume*slider_multiplier_max), startvalue=model.total_volume)
		volume_slider_value = Label(slider_space[3,2], string(round(volume_slider.value[], digits=2)) * " m³")

		n_mol_slider_label = Label(slider_space[4,0], "Teilchen: ", fontsize=16)
		n_mol_slider = Slider(slider_space[4,1], range = model.n_mol*slider_multiplier_min:model.n_mol/slider_step_quotient:model.n_mol*slider_multiplier_max, startvalue=model.n_mol)
		n_mol_slider_value = Label(slider_space[4,2], string(round(n_mol_slider.value[], digits=2)) * " mol")


		on(abmobs.model) do _
			e_internal_label.text[] = string("Eᵢ: ", string(round(model.e_internal)), " J")
			#pressure_label.text[] = string("Druck: ", string(round(model.pressure_bar, digits=2)), " Bar")
			if model.mass_gas > 999.9
				mass_label.text[] = string("Masse: ", string(round(model.mass_gas/1000, digits=3), " kg"))
			else
				mass_label.text[] = string("Masse: ", string(model.mass_gas), " g")
			end
			#volume_label.text[] = string("Volumen: " * string(round(model.total_volume, digits=2))* " m³ ; " * string(round(model.total_volume * 1000, digits=2)) * " L")

		end

		on(gas_dropdown.selection) do selected_gas
			new_molarmass = model.gases[selected_gas]
			model.molar_mass = new_molar_mass
			model.mass_kg = new_molar_mass * 1.66054e-27
			model.mass_gas = round(model.n_mol * model.molar_mass, digits=3)
			
		end

		on(mode_dropdown.selection) do selected_mode
			model.mode = model.modes[selected_mode]
		end

		# on change of temperature slider
		on(temp_slider.value) do temp
			if model.mode == "temp-druck" || model.mode == "temp-vol"
				temp_slider_value.text[] = string(round(temp[], digits=2)) * " K"
				model.temp = temp[]
			end
			
			# If the mode is "temp-druck" the pressure is calculated
			if model.mode == "temp-druck"
				pressure = calc_pressure(model)
				pressure_slider_pa_value.text[] = string(round(pressure, digits=0)) * " Pa"
				model.pressure_pa = pressure
				set_close_to!(pressure_slider_pa, pressure)
				pressure_slider_bar_value.text[] = string(round(pressure / 1e5, digits=2)) * " Bar"
				model.pressure_bar = pressure / 1e5
				set_close_to!(pressure_slider_bar, pressure / 1e5)

				# set the range of the sliders
				pressure_slider_pa.range = pressure * slider_multiplier_min:pressure/slider_step_quotient:pressure * slider_multiplier_max
				pressure_slider_bar.range = pressure / 1e5 * slider_multiplier_min:pressure / 1e5 / slider_step_quotient:pressure / 1e5 * slider_multiplier_max

			# If the mode is "temp-vol" the volume is calculated
			elseif model.mode == "temp-vol"
				volume = model.n_mol * 8.314 * model.temp/ model.pressure_pa
				volume_slider_value.text[] = string(round(volume, digits=2)) * " m³"
				set_close_to!(volume_slider, volume)

				# set the range of the sliders
				volume_slider.range = model.volume * slider_multiplier_min:model.volume/slider_step_quotient:model.volume * slider_multiplier_max
			end
		end

		# on change of pressure bar slider
		on(pressure_slider_bar.value) do pressure
			if model.mode == "druck-vol" || model.mode == "druck-temp"
				pressure_slider_bar_value.text[] = string(round(pressure[], digits=2)) * " Bar"
				model.pressure_bar = pressure[]

				pressure_slider_pa_value.text[] = string(round(pressure[] * 1e5, digits=0)) * " Pa"
				set_close_to!(pressure_slider_pa, round(pressure[] * 1e5), digits=0)
				model.pressure_pa = pressure[] * 1e5
				
				# If the mode is "druck-vol" the volume is calculated
				if model.mode == "druck-vol"
					volume = model.n_mol * 8.314 * model.temp/ model.pressure_pa
					volume_slider_value.text[] = string(round(volume, digits=2)) * " m³"
					set_close_to!(volume_slider, volume)

					# set the range of the sliders
					volume_slider.range = model.volume * slider_multiplier_min:model.volume/slider_step_quotient:model.volume * slider_multiplier_max

				# If the mode is "druck-temp" the temperature is calculated
				elseif model.mode == "druck-temp"
					temp = calc_temperature(model)
					temp_slider_value.text[] = string(round(temp, digits=2)) * " K"
					set_close_to!(temp_slider, temp)

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max

				end

			end
		end

		# on change of pressure pa slider
		on(pressure_slider_pa.value) do pressure
			if model.mode == "druck-vol" || model.mode == "druck-temp"
				pressure_slider_pa_value.text[] = string(round(pressure[], digits=0)) * " Pa"
				model.pressure_pa = pressure[]

				pressure_slider_bar_value.text[] = string(round(pressure[] / 1e5, digits=2)) * " Bar"
				set_close_to!(pressure_slider_bar, round(pressure[] / 1e5), digits=2)
				model.pressure_bar = pressure[] / 1e5

				# If the mode is "druck-vol" the volume is calculated
				if model.mode == "druck-vol"
					volume = model.n_mol * 8.314 * model.temp/ model.pressure_pa
					volume_slider_value.text[] = string(round(volume, digits=2)) * " m³"
					set_close_to!(volume_slider, volume)

					# set the range of the sliders
					volume_slider.range = model.volume * slider_multiplier_min:model.volume/slider_step_quotient:model.volume * slider_multiplier_max

				# If the mode is "druck-temp" the temperature is calculated
				elseif model.mode == "druck-temp"
					temp = calc_temperature(model)
					temp_slider_value.text[] = string(round(temp, digits=2)) * " K"
					set_close_to!(temp_slider, temp)

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max

				end

			end
		end

		# on change of volume slider
		on(volume_slider.value) do volume
			if model.mode == "vol-druck" || model.mode == "vol-temp"
				volume_slider_value.text[] = string(round(volume[], digits=2)) * " m³"
				model.total_volume = volume[]

				# If the mode is "vol-druck" the pressure is calculated
				if model.mode == "vol-druck"
					pressure = calc_pressure(model)
					pressure_slider_pa_value.text[] = string(round(pressure, digits=0)) * " Pa"
					model.pressure_pa = pressure[]
					set_close_to!(pressure_slider_bar, pressure[] / 1e5)
					pressure_slider_bar_value.text[] = string(round(pressure / 1e5, digits=2)) * " Bar"
					model.pressure_bar = pressure / 1e5
					set_close_to!(pressure_slider_pa, pressure)

					# set the range of the sliders
					pressure_slider_pa.range = pressure * slider_multiplier_min:pressure/slider_step_quotient:pressure * slider_multiplier_max
					pressure_slider_bar.range = pressure / 1e5 * slider_multiplier_min:pressure / 1e5 / slider_step_quotient:pressure / 1e5 * slider_multiplier_max

				# If the mode is "vol-temp" the temperature is calculated
				elseif model.mode == "vol-temp"
					temp = calc_temperature(model)
					temp_slider_value.text[] = string(round(temp, digits=2)) * " K"
					set_close_to!(temp_slider, temp)

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max

				end
			end
		end

		# on change of n_mol slider
		on(n_mol_slider.value) do n_mol
			if model.mode == "mol-temp" || model.mode == "mol-druck"
				n_mol_slider_value.text[] = string(round(n_mol[], digits=2)) * " mol"
				model.n_mol = n_mol[]
				model.n_particles = model.n_mol * 6.022e23 / 1e22 / 8

				# If the number of particles has changed, agents are added or removed
				if model.n_particles_old < model.n_particles

					for _ in model.n_particles_old:model.n_particles
						vel = Tuple( 2rand(2).-1)
						vel = vel ./ norm(vel)  # ALWAYS maintain normalised state of vel!
						speed = sqrt((3 * R * model.temp) / model.molar_mass / 1000)  # Initial speed based on temperature (molar_mass converted to kg/mol)
						speed = scale_speed(speed, model.max_speed)  				# Scale speed to avoid excessive velocities
						add_agent!( model, vel, model.mass_kg, speed, model.radius, non_id, -Inf)
					end
			
					model.n_particles_old = model.n_particles
			
				elseif model.n_particles_old > model.n_particles
			
					for _ in model.n_particles:model.n_particles_old
						agent = random_agent(model)
						kill_agent!(agent, model)
					end
			
					model.n_particles_old = model.n_particles
				end

				# If the mode is "n_mol-temp" the temperature is calculated
				if model.mode == "mol-temp"
					temp = calc_temperature(model)
					temp_slider_value.text[] = string(round(temp, digits=2)) * " K"
					set_close_to!(temp_slider, temp)

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max


				# If the mode is "n_mol-druck" the pressure is calculated
				elseif model.mode == "mol-druck"
					pressure = calc_pressure(model)
					pressure_slider_pa_value.text[] = string(round(pressure, digits=0)) * " Pa"
					model.pressure_pa = pressure[]
					set_close_to!(pressure_slider_bar, pressure[] / 1e5)
					pressure_slider_bar_value.text[] = string(round(pressure / 1e5, digits=2)) * " Bar"
					model.pressure_bar = pressure / 1e5
					set_close_to!(pressure_slider_pa, pressure)

					# set the range of the sliders
					pressure_slider_pa.range = pressure * slider_multiplier_min:pressure/slider_step_quotient:pressure * slider_multiplier_max
					pressure_slider_bar.range = pressure / 1e5 * slider_multiplier_min:pressure / 1e5 / slider_step_quotient:pressure / 1e5 * slider_multiplier_max
				end
			end
		end


		playground
	end

	function increase_vol_const(i::Int = 1)
		print("increase volume constant")# * i)
	end

	function decrease_vol_const(i::Int = 1)
		print("decrease_vol_const") #* i)
	end 
	

end	# of module IdealGas
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
	gases = Dict("Helium" => 4.0, "Hydrogen" => 2.0, "Oxygen" => 32.0),					# Gas types
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
	temp_old 				= 293.15,															# Temperature of the previous step in Kelvin
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
	old_scaled_speed 		= 0.0,																# Scaled speed of the previous step
	step 					= 0,																# Step counter
	max_speed 				= 8000.0,  															# Maximum speed of the particles in m/s
	extent 					= (500,500),														# Extent of Particles space
)
    space = ContinuousSpace(extent; spacing = 2.5)												# Create the space for the particles

	properties = Dict(
		:n_particles	=> n_particles,
		:temp		=> temp,
		:temp_old		=> temp_old,
		:total_volume	=> total_volume,
		:e_internal	=> e_internal,
		:old_scaled_speed => old_scaled_speed,
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

	scaled_speed = calc_and_scale_speed(model)
	model.old_scaled_speed = scaled_speed
	for _ in 1:n_particles
		vel = Tuple( 2rand(2).-1)
		vel = vel ./ norm(vel)  # ALWAYS maintain normalised state of vel!
        add_agent!( model, vel, mass_kg, scaled_speed, radius, non_id, -Inf)
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

	model.entropy_change = calc_entropy_change(model)
	model.e_internal = calc_internal_energy(model)

	scaled_speed = calc_and_scale_speed(model)
	# Set speed of all particles the same when approaching absolute zero (Downwards from T=60 K)
	if scaled_speed < 1
		for particle in allagents(model)
			particle.speed = scaled_speed
		end
	else # Else keep the difference in speed to the root mean square speed of the previous step (but scale it for visuals)
		for particle in allagents(model)
			particle.speed = scaled_speed + (particle.speed - model.old_scaled_speed) / (7/4)
		end
	end
	model.old_scaled_speed = scaled_speed

	model.step += 1.0

end

#----------------------------------------------------------------------------------------

"""
	demo()

Run a simulation of the IdealGas model.
"""



	function demo()
		
		function set_slider(value, slider, slider_value, unit)
			slider_value.text[] = string(round(value, digits=2), " ", unit)
			set_close_to!(slider, round(value, digits=2))
		end

		function add_or_remove_agents!(model)
			if model.n_particles_old < model.n_particles
				scaled_speed = calc_and_scale_speed(model)
				for _ in model.n_particles_old:model.n_particles
					vel = Tuple( 2rand(2).-1)
					vel = vel ./ norm(vel)  # ALWAYS maintain normalised state of vel!
					add_agent!( model, vel, model.mass_kg, scaled_speed, model.radius, non_id, -Inf)
				end
		
				model.n_particles_old = model.n_particles
		
			elseif model.n_particles_old > model.n_particles
		
				for _ in model.n_particles:model.n_particles_old
					agent = random_agent(model)
					kill_agent!(agent, model)
				end
		
				model.n_particles_old = model.n_particles
			end
		end

		function create_custom_slider(slider_space, row_num, labeltext, fontsize, range, unit, startvalue)
			label = Label(slider_space[row_num, 0], labeltext, fontsize=fontsize)
			slider = Slider(slider_space[row_num, 1], range=range, startvalue=startvalue)
			slider_value = Label(slider_space[row_num, 2], string(slider.value[]) * " " * unit)
			return label, slider, slider_value
		end
		
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

		temp_slider_label, temp_slider, temp_slider_value = create_custom_slider(slider_space, 0, "Temperatur: ", 16, model.temp*slider_multiplier_min:model.temp/slider_step_quotient:model.temp*slider_multiplier_max, "K", model.temp)
		pressure_slider_bar_label, pressure_slider_bar, pressure_slider_bar_value = create_custom_slider(slider_space, 1, "Druck[Bar]: ", 16, model.pressure_bar*slider_multiplier_min:model.pressure_bar/slider_step_quotient:model.pressure_bar*slider_multiplier_max, "Bar", model.pressure_bar)
		pressure_slider_pa_label, pressure_slider_pa, pressure_slider_pa_value = create_custom_slider(slider_space, 2, "Druck[Pa]: ", 16, model.pressure_pa*slider_multiplier_min:model.pressure_pa/slider_step_quotient:model.pressure_pa*slider_multiplier_max, "Pa", model.pressure_pa)
		volume_slider_label, volume_slider, volume_slider_value = create_custom_slider(slider_space, 3, "Volumen: ", 16, model.total_volume*slider_multiplier_min:model.total_volume/slider_step_quotient:model.total_volume*slider_multiplier_max, "m³", model.total_volume)
		n_mol_slider_label, n_mol_slider, n_mol_slider_value = create_custom_slider(slider_space, 4, "Stoffmenge: ", 16, model.n_mol*slider_multiplier_min:model.n_mol/slider_step_quotient:model.n_mol*slider_multiplier_max, "mol", model.n_mol)

		is_updating = false

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
				model.temp = temp[]
				temp_slider_value.text[] = string(round(temp[], digits=2)) * " K"
			end
			
			# If the mode is "temp-druck" the pressure is calculated
			if model.mode == "temp-druck"
				model.pressure_pa = calc_pressure(model)
				model.pressure_bar = model.pressure_pa / 1e5
				set_slider(model.pressure_pa, pressure_slider_pa, pressure_slider_pa_value, "Pa")
				set_slider(model.pressure_bar, pressure_slider_bar, pressure_slider_bar_value, "Bar")

				# set the range of the sliders
				pressure_slider_pa.range = model.pressure_pa * slider_multiplier_min:model.pressure_pa/slider_step_quotient:model.pressure_pa * slider_multiplier_max
				pressure_slider_bar.range = model.pressure_bar * slider_multiplier_min:model.pressure_bar / slider_step_quotient:model.pressure_bar * slider_multiplier_max

			# If the mode is "temp-vol" the volume is calculated
			elseif model.mode == "temp-vol"
				model.total_volume = calc_volume(model)
				set_slider(model.total_volume, volume_slider, volume_slider_value, "m³")
				model.volume = calc_total_vol_dimension(model.total_volume)

				# set the range of the sliders
				volume_slider.range = model.total_volume * slider_multiplier_min:model.total_volume/slider_step_quotient:model.total_volume * slider_multiplier_max
			end
		end

		# on change of pressure bar slider
		on(pressure_slider_bar.value) do pressure
			if is_updating
				return
			end
			is_updating = true
			if model.mode == "druck-vol" || model.mode == "druck-temp"
				model.pressure_bar = pressure[]
				model.pressure_pa = pressure[] * 1e5
				pressure_slider_bar_value.text[] = string(round(model.pressure_bar, digits=2)) * " Bar"
				set_slider(model.pressure_pa , pressure_slider_pa, pressure_slider_pa_value, "Pa")
				
				# If the mode is "druck-vol" the volume is calculated
				if model.mode == "druck-vol"
					model.total_volume = calc_volume(model)
					set_slider(model.total_volume, volume_slider, volume_slider_value, "m³")
					model.volume = calc_total_vol_dimension(model.total_volume)

					# set the range of the sliders
					volume_slider.range = model.total_volume * slider_multiplier_min:model.total_volume/slider_step_quotient:model.total_volume * slider_multiplier_max
				# If the mode is "druck-temp" the temperature is calculated
				elseif model.mode == "druck-temp"
					model.temp = calc_temperature(model)
					set_slider(model.temp, temp_slider, temp_slider_value, "K")

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max

				end
			end
			is_updating = false
		end

		# on change of pressure pa slider
		on(pressure_slider_pa.value) do pressure
			if is_updating
				return
			end
			is_updating = true
			if model.mode == "druck-vol" || model.mode == "druck-temp"
				model.pressure_pa = pressure[]
				model.pressure_bar = pressure[] / 1e5
				pressure_slider_pa_value.text[] = string(round(model.pressure_pa, digits=2)) * " Pa"
				set_slider(model.pressure_bar, pressure_slider_bar, pressure_slider_bar_value, "Bar")

				# If the mode is "druck-vol" the volume is calculated
				if model.mode == "druck-vol"
					model.total_volume = calc_volume(model)
					set_slider(model.total_volume, volume_slider, volume_slider_value, "m³")
					model.volume = calc_total_vol_dimension(model.total_volume)

					# set the range of the sliders
					volume_slider.range = model.total_volume * slider_multiplier_min:model.total_volume/slider_step_quotient:model.total_volume * slider_multiplier_max
				# If the mode is "druck-temp" the temperature is calculated
				elseif model.mode == "druck-temp"
					model.temp = calc_temperature(model)
					set_slider(model.temp, temp_slider, temp_slider_value, "K")

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max

				end
			end
			is_updating = false
		end

		# on change of volume slider
		on(volume_slider.value) do volume
			if model.mode == "vol-druck" || model.mode == "vol-temp"
				model.total_volume = volume[]
				volume_slider_value.text[] = string(round(volume[], digits=2)) * " m³"
				model.volume = calc_total_vol_dimension(model.total_volume)

				# If the mode is "vol-druck" the pressure is calculated
				if model.mode == "vol-druck"
					model.pressure_pa = calc_pressure(model)
					model.pressure_bar = model.pressure_pa / 1e5
					set_slider(pressure, pressure_slider_pa, pressure_slider_pa_value, "Pa")
					set_slider(pressure / 1e5, pressure_slider_bar, pressure_slider_bar_value, "Bar")

					# set the range of the sliders
					pressure_slider_pa.range = model.pressure_pa * slider_multiplier_min:model.pressure_pa/slider_step_quotient:model.pressure_pa * slider_multiplier_max
					pressure_slider_bar.range = model.pressure_bar * slider_multiplier_min:model.pressure_bar / slider_step_quotient:model.pressure_bar * slider_multiplier_max

				# If the mode is "vol-temp" the temperature is calculated
				elseif model.mode == "vol-temp"
					model.temp = calc_temperature(model)
					set_slider(model.temp, temp_slider, temp_slider_value, "K")

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max

				end
			end
		end

		# on change of n_mol slider
		on(n_mol_slider.value) do n_mol
			if model.mode == "mol-temp" || model.mode == "mol-druck"
				model.n_mol = n_mol[]
				n_mol_slider_value.text[] = string(round(n_mol[], digits=2)) * " mol"
				model.n_particles = model.n_mol * 6.022e23 / 1e22 / 8
				add_or_remove_agents!(model)				

				# If the mode is "n_mol-temp" the temperature is calculated
				if model.mode == "mol-temp"
					model.temp = calc_temperature(model)
					set_slider(model.temp, temp_slider, temp_slider_value, "K")

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max


				# If the mode is "n_mol-druck" the pressure is calculated
				elseif model.mode == "mol-druck"
					model.pressure_pa = calc_pressure(model)
					model.pressure_bar = model.pressure_pa / 1e5
					set_slider(model.pressure_pa, pressure_slider_pa, pressure_slider_pa_value, "Pa")
					set_slider(model.pressure_bar, pressure_slider_bar, pressure_slider_bar_value, "Bar")

					# set the range of the sliders
					pressure_slider_pa.range = model.pressure_pa * slider_multiplier_min:model.pressure_pa/slider_step_quotient:model.pressure_pa * slider_multiplier_max
					pressure_slider_bar.range = model.pressure_bar * slider_multiplier_min:model.pressure_bar / slider_step_quotient:model.pressure_bar * slider_multiplier_max
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
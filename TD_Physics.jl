#========================================================================================#
"""
TD_Physics

Costum Library of ThermoDynamics for IdealGas-simulation.

Author: Francisco Hella, Felix Rollbühler, Melanie *, Jan Wiechmann, 20/06/23
"""

module TD_Physics

include("AgentTools.jl")
using Agents, LinearAlgebra, GLMakie, InteractiveDynamics, GeometryBasics, Observables

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

end #------------------------------------------------------------------------------------------

# Ideal-Gas-Project

***This simulation will provide first-year engineering undergraduates with  hands-on experience of ideal gases. Users explore the behaviour of a closed ideal gas system by setting its thermodynamic state variables/constants (e.g., 𝑝, 𝑉, 𝑇, 𝐸ᵢ, Δ𝑆), applying work  and/or heat to the system.***

<hr>


<sub><td>The following video displays the simulation of **ideal gas behaviour** depending on the **system variables**</td></sub> <br>
<sub><td>* pressure **p**</td></sub><br>
<sub><td>* volume **V**</td></sub><br>
<sub><td>* temperature **T**</td></sub><br>
<sub><td>* internal energy **𝐸ᵢ**</td></sub><br>
<sub><td>* entropy change **Δ𝑆**</td></sub><br>




https://github.com/melanie-s-h/Ideal-Gas-Projekt/assets/134691659/b1b79bc2-4213-47bf-8091-8b954f52ccb2

*The Application was build in collaboration with three other students, the parts implemented by me are as follows:*
<br>
* **Root-mean-square speed:** Calculated the root-mean-square speed of the particles based on temperature or (temperature change) of the gas and scaled the result to match the visual demonstration and to avoid excessive velocities.
    * *uᵣₘₛ = sqrt(3*R*T / M)*
      * R: Gas Constant
      * T: Temperature
      * M: Molar mass of gas
* **Speed Redistribution:** Redistributing the speed each step of the simulation of each particle to match the visual representation of the calculated root-mean-square speed.

* **Internal Energy:** Calculating the internal energy of the ideal gas based on the temperature each steap of the model based on the changed parameters.
   * *Eᵢ = f * 1/2 * n * R * T*
      * Eᵢ: Internal Energy
      * f: Degrees of Freedom (of the Particles)
      * n: Mol of gas
        
* **Change of entropy:** Calculating the specific heat capacity and the change in entropy depending on the thermodynamic process happening at the moment. Displaying the entropy change in the simulation GUI in a diagram.
   * For isochoric process or |isochoric & isothermal|-process
        * *Specific heat capacity cₚ = (f+2) * R/2 * n / m*
             * m: Mass of gas
        * *Change in entropy Δs = cₚ · ln(T₂/T₁) + Rᵢ · ln(p₂/p₁)*
   * For isobaric process or isothermal process
        * *Specific heat capacity cᵥ = f * R/2 * n / m*
        * *Change in entropy Δs = cᵥ · ln(T₂/T₁) + Rᵢ · ln(V₂/V₁)*
* Rework of the graphical user interface and the plotting kwargs.
* Fitting the size of the simulation-UI dynamically to the display size of the end user.
* Prevention of polytropic thermodynamic processes by halting the volume change in non-volume relevant modes.

<br>
<br>
<hr>
<br>

    
<sub><td>**Running the Application:** *(Instructions in German)* </td></sub><br>
<sub><td>Starten des Programms:</td></sub>
<sub><td><br>1. Installieren von Julia</td></sub>
<sub><td><br>2. Installieren der nötigen Packages</td></sub>
<sub><td><br>3. Julia Konsole öffnen</td></sub>
         <br><sub><td>&nbsp;* Using Pkg</td></sub>
         <br><sub><td>&nbsp;* Pkg.add("Paketname@vX.Y.Z")</td></sub>
         <br> <sub><td>&nbsp;&nbsp;- Agents v5.14.0</td></sub>
        <br><sub><td>&nbsp;&nbsp;- LinearAlgebra</td></sub>
        <br><sub><td>&nbsp;&nbsp;- GLMakie v0.8.5</td></sub>
        <br><sub><td>&nbsp;&nbsp;- InteractiveDynamics v0.22.1</td></sub>
        <br><sub><td>&nbsp;&nbsp;- Observables v0.5.4</td></sub>
<br><sub><td>4. Mit dem "cd()" command zu dem Verzeichnis navigieren, indem "IdealGas.jl" enthalten ist</td></sub>
<br><sub><td>5. Include("IdealGas.jl")</td></sub>
<br><sub><td>6. IdealGas.demo()</td></sub>


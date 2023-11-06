# Ideal-Gas-Project
<br>

***This simulation will provide first-year engineering undergraduates with  hands-on experience of ideal gases. Users explore the behaviour of a closed ideal gas system by setting its thermodynamic state variables/constants (e.g., 𝑝, 𝑉, 𝑇, 𝐸𝐼 , 𝑆), applying work  and/or heat to the system.***

<hr>
<br>


https://github.com/melanie-s-h/Ideal-Gas-Projekt/assets/134691659/b1b79bc2-4213-47bf-8091-8b954f52ccb2

*The Application was build in collaboration with three other students, the parts implemented by me are as follows:*
<br>
* **Root-mean-square speed:** Calculated the root mean squared speed of the particles based on temperature of the gas and scaled the result to match the visual demonstration.
    * *uᵣₘₛ = sqrt(3*R*T / M)* 


<br>
<hr>
<br>
**Running the Application:** *(Instructions in German)*
Starten des Programms:
1. installieren von Julia
2. installieren der nötigen Packages
    1. Julia Konsole öffnen
    2. using Pkg
    3. Pkg.add("Paketname@vX.Y.Z")
        - Agents v5.14.0
        - LinearAlgebra
        - GLMakie v0.8.5
        - InteractiveDynamics v0.22.1
        - Observables v0.5.4
3. Mit dem "cd()" command zu dem Verzeichnis navigieren, indem "IdealGas.jl" enthalten ist
4. include("IdealGas.jl")
5. IdealGas.demo()

# Passivity Enforcement
### Author: Brian Guiana
### Date: 5/14/2021

This code was written for a class project in the course entitled ECE 504: "ST: Passive Electromagnetic Systems" taught in Spring 2021 by Dr. Ata Zadehgol at the University of Idaho in Moscow. 

This code was developed, in part, based on the code developed by Jennifer Houle in ECE 504 "ST: Modern Circuit Synthesis Algorithms" taught in Spring 2020 by Dr. Ata Zadehgol. Jennifer's code is available online at https://github.com/JenniferEHoule/Circuit_Synthesis.


# Overview
- This code implements the passivity enforcement algorithm detailed in [11]
- This code uses programs written in [12] in order to get a vector fit state-space model to enforce passivity onto
- This code was intended to be able to replace the file RPdriver.py in [12] for passivity enforcement

# Licensing
From [7], restrictions on use, in addition to licensing:

>- Embedding any of (or parts from) the routines of the Matrix Fitting Toolbox in a commercial software, or a software requiring licensing, is strictly prohibited. This applies to all routines, see Section 2.1.
>- If the code is used in a scientific work, then reference should me made as follows:
>  - VFdriver.m and/or vectfit3.m: References [1],[2],[3]
>  - RPdriver.m and/or FRPY.m applied to Y-parameters: [8],[9]

# Files
## Primary Files, [12] was the original creator of each associated file.
- create_netlist.py [12]
- passivity_driver.py [11]
- plots.py [12]
- utils.py [12]
- vectfit3.py [12]
- VFdriver.py [12]

## Secondary Files
- ex4_Y.py: 4-port example file
- input_4_port_ads.s4p: Input for ex4_y.py
- netlist.sp: Output for ex4_y.py

# Inputs
- Singular .sNp file. Vector fitting is applied to this file followed by passivity enforcement

# Outputs
- Singular .sp file. PSPICE simulation file for use with SPICE simulation software.

# References
```
[1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
    domain responses by Vector Fitting", IEEE Trans. Power Delivery,
    vol. 14, no. 3, pp. 1052-1061, July 1999.

[2] B. Gustavsen, "Improving the pole relocating properties of vector
    fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
    July 2006.

[3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
    "Macromodeling of Multiport Systems Using a Fast Implementation of
    the Vector Fitting Method", IEEE Microwave and Wireless Components
    Letters, vol. 18, no. 6, pp. 383-385, June 2008.

[4] B. Gustavsen, VFIT3, The Vector Fitting Website. March 20, 2013. Accessed on:
    Jan. 21, 2020. [Online]. Available: 
    https://www.sintef.no/projectweb/vectfit/downloads/vfit3/.

[5] A. Zadehgol, "A semi-analytic and cellular approach to rational system characterization 
    through equivalent circuits", Wiley IJNM, 2015. [Online]. https://doi.org/10.1002/jnm.2119

[6] V. Avula and A. Zadehgol, "A Novel Method for Equivalent Circuit Synthesis from 
    Frequency Response of Multi-port Networks", EMC EUR, pp. 79-84, 2016. [Online]. 
    Available: ://WOS:000392194100012.

[7] B. Gustavsen, Matrix Fitting Toolbox, The Vector Fitting Website.
    March 20, 2013. Accessed on: Feb. 25, 2020. [Online]. Available:
    https://www.sintef.no/projectweb/vectorfitting/downloads/matrix-fitting-toolbox/.

[8] B. Gustavsen, "Fast passivity enforcement for S-parameter models by perturbation
    of residue matrix eigenvalues",
    IEEE Trans. Advanced Packaging, vol. 33, no. 1, pp. 257-265, Feb. 2010.

[9] B. Gustavsen, "Fast Passivity Enforcement for Pole-Residue Models by Perturbation
    of Residue Matrix Eigenvalues", IEEE Trans. Power Delivery, vol. 23, no. 4,
    pp. 2278-2285, Oct. 2008.

[10] A. Semlyen, B. Gustavsen, "A Half-Size Singularity Test Matrix for Fast and Reliable
    Passivity Assessment of Rational Models," IEEE Trans. Power Delivery, vol. 24, no. 1,
    pp. 345-351, Jan. 2009.

[11] E. Medina, A Ramirez, J. Morales, and K. Sheshyekani, “Passivity Enforcement of FDNEs via Perturbation of Singularity Test Matrix,” IEEE Trans. on Power Del., vol. 35, no. 4, pp. 1648-1655, Aug. 2020.

[12] JenniferEHoule, “Circuit_Synthesis,” GitHub, May 12, 2020, Accessed: Mar. 10, 2021, Online, Available: https://github.com/JenniferEHoule/Circuit_Synthesis



```

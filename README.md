# LLPS-ProteinRNA_Condensates-CP2P2024
# Setting up Development Environment (02-09)
- devcontainer.json file is created: using Python 3.9
- **Title**: Focusing on the Liquid-Liquid Phase Separation (LLPS) of Protein-RNA condensates.
- **Code**: Establish a comprehensive and flexible Python-based framework for the detailed study of LLPS in Protein-RNA condensates, which aims to facilitate the computational analysis of experimental data.
- **Tools**: This includes using GitHub for version control, setting up GitHub Actions for CI, incorporating linting tools (pylint), and establishing a framework for unit testing (pytest) to ensure the project's sustainability and ease of contribution.

# Data Types

This section of the documentation details the data types used in the project, with a focus on their representations in Python, specifically using Python 3.9 annotations.

## Overview

The project involves processing and analyzing numerical data represented in a structured format. Each piece of data comprises several floating-point values, each serving a unique purpose in the dataset.

## Sample Data Format

Here is an example (as a sample) of the data format (first five rows), floating-point numbers:
```
1.000000000000000021e-02	2.625600000000000069e-05	4.446704900000000155e-02	1.000000000000000021e-02	3.700092800000000237e-02	-4.825906299999999788e-02	4.446704945999999659e+00
1.035261699999999972e-02	2.814100000000000043e-05	4.603503400000000245e-02	1.035261699999999972e-02	3.426594799999999746e-02	-4.904761400000000326e-02	4.446704829999999831e+00
1.071766900000000085e-02	3.015999999999999923e-05	4.765830900000000286e-02	1.071766900000000085e-02	3.137005799999999928e-02	-4.977367299999999745e-02	4.446704705000000146e+00
1.109559300000000082e-02	3.232499999999999968e-05	4.933882199999999746e-02	1.109559300000000082e-02	2.830639399999999861e-02	-5.042681899999999773e-02	4.446704570999999717e+00
1.148684300000000005e-02	3.464400000000000238e-05	5.107859399999999800e-02	1.148684300000000005e-02	2.506817799999999999e-02	-5.099564100000000127e-02	4.446704428000000320e+00
```

## Data Types in Python

### Floating-Point Numbers

The primary data type used in this project is the floating-point number, represented by `float` in Python. This choice is due to the nature of the data, which includes real numbers with decimal points, necessitating precision in calculations and data representation.

### Data Structure

A single row of data is represented as a tuple of floats, encapsulating the structured nature of the dataset. This approach allows for the efficient and clear handling of data points throughout the project's codebase.

#### Python Annotation

```python
from typing import List  # Correct import but unnecessary for this example being built-in for Python 3.9 or later.

# Define a type alias for a data point
DataPoint = list[float]
```

# Random Numbers
In this investigation, we utilized video particle tracking, specifically employing Trackmate software, to accurately determine the diffusion coefficient of an 80% glycerol solution. This objective was achieved by precisely tracking the trajectory of 1-micrometer beads displaying Brownian motion across 1000 video frames, facilitated by the use of EPI fluorescence microscopy for enhanced observation. This methodical approach not only enables a detailed analysis of the bead movements but also sheds light on the underlying diffusion processes within the glycerol solution.

Random walk, observed experimentally, is used to analyze the motion of tracked bead  in a microscopy image and characterize their motion through Mean Squared Displacement (MSD) analysis.  

The Figure 1 represents the Gaussian distribution of displacements for a particle undergoing a random walk, at selected time point. The concept of a random walk, often used to describe the motion of particles in glycerol solution (Brownian motion), pertains to the particle moving in random directions at each step, with the displacement from the origin evolving over time. In the figure, the x-axis shows the possible displacements of the particle from its starting point. Displacements to the left of the origin indicate movement in one direction, while displacements to the right indicate movement in the opposite direction. The y-axis represents the probability density of finding the bead at a given displacement. The curve is Gaussian distribution, characterized by its mean $\mu$ and standard deviation $\sigma$


The curve is centered around zero displacement, indicating that, on average, the particle is expected to be at its starting point. This is a hallmark of a symmetric random walk, where the particle has an equal chance of moving in any direction. The relationship $\(\sigma = \sqrt{2Dt}\)$ shows that the standard deviation grows as the square root of time, assuming a constant diffusion coefficient \(D\). This increase in spread reflects the increasing uncertainty in the particle's position over time. The specific form of the distributions (Gaussian) and their evolution over time suggest normal diffusion, where the mean squared displacement is proportional to time $\(\text{MSD} = 4Dt\)$.


### Python Code: Gaussian Distribution (Brownian Motion) 

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Extract the diffusion constant D from the fitted parameters
D = 0.14 # Obtained experimentally, D = 0.14161575 +/- 0.00148164: Diffusion Coefficient of Glycerol taken as a sample

# Choose several time points to plot the Gaussian distribution of displacements
time_point = 100 

# Compute the standard deviation for the current time point
sigma = np.sqrt(2 * D * t)
    
# Generate a range of displacements around 0
displacements = np.linspace(-3*sigma, 3*sigma, 1000)
    
# Compute the probability density function (PDF) for the displacements
pdf = norm.pdf(displacements, 0, sigma)
    
# Plot the PDF
plt.plot(displacements, pdf, label=f't={t}s')

plt.xlabel('Displacement')
plt.ylabel('Probability Density')
plt.title('Gaussian Distribution of Displacements')
plt.legend()
plt.show()
# Save the plot with high resolution
plt.savefig('Gaussian.png', dpi=300)
```


![Gaussian Distribution Enhanced](https://github.com/ubsuny/LLPS-ProteinRNA_Condensates-CP2P2024/assets/143649367/b6e0e455-edf9-4e51-ae1d-92eb2a5a85ee)

*Figure 1 - Gaussian distribution of displacements for a random walk.*

# Viscosity Measurement in Protein-DNA Condensates Using Video Particle Tracking (VPT)

## Introduction to the Physics of the Project

The phenomenon of Liquid-Liquid Phase Separation (LLPS) is a critical biophysical process that underlies the formation of biomolecular condensates within cells. These condensates are essential for various cellular functions, including gene expression regulation, signal transduction, and stress response. LLPS involves the demixing of a homogenous solution into two distinct phases: a dense phase (condensate) and a dilute phase. This process is driven by multivalent interactions among biomolecules, such as proteins and nucleic acids, leading to the spontaneous organization of intracellular matter.

Understanding the material properties, such as viscosity, of these condensates is crucial for elucidating their functional roles within the cellular environment. Viscosity, a measure of a fluid's resistance to flow, influences the rate of molecular exchange within and between the condensates and the surrounding medium. Therefore, accurately quantifying the viscosity of protein-DNA condensates is fundamental to comprehending their dynamics and functions.

## Literature Review

### Background

The article "Quantifying viscosity and surface tension of multicomponent protein-nucleic acid condensates" by Ibraheem Alshareedah, George M. Thurston, and Priya R. Banerjee presents an experimental framework to quantify the physical properties of biomolecular condensates. Our project aims to reproduce a part of their study focusing on measuring the viscosity of protein-DNA condensates using the Video Particle Tracking (VPT) method.

### Objective

The primary objective of this project is to employ VPT to estimate the viscosity of protein-DNA condensates formed through LLPS. This method involves tracking the motion of tracer particles embedded within the condensates and analyzing their movement to derive viscosity values.

### Methodology

1. **Sample Preparation**: Create protein-DNA condensates in vitro using a model system, such as the positively charged low-complexity disordered polypeptide $[RGRGG]_5$ and a negatively charged single-stranded DNA (ssDNA: dT40). Mix these components in a buffer solution at varying stoichiometries to induce phase separation.

2. **Embedding Tracer Particles**: Introduce fluorescently labeled polystyrene beads of a known size (e.g., 200 nm) into the mixture as tracer particles before the onset of phase separation.

3. **Video Particle Tracking**: Utilize a fluorescence microscope equipped with a high-speed camera to capture real-time videos of the tracer particles moving within the condensates. Ensure to record multiple videos covering different regions of the sample to gather a statistically significant dataset.

4. **Data Analysis**:
   - Extract the trajectories of individual tracer particles from the videos using image analysis software.
   - Calculate the Mean Squared Displacement (MSD) of the tracer particles over time.
   - Apply the Stokes-Einstein relation to derive the viscosity of the condensates from the MSD data.

### Expected Outcomes

- The viscosity measurements should yield results consistent with those reported in the literature, within the experimental error margins.
- The viscosity values are anticipated to provide insights into the internal dynamics of the protein-DNA condensates and how they are influenced by factors such as mixture stoichiometry.

### Conclusion

By focusing on the VPT method to estimate the viscosity of protein-DNA condensates, this project aims to contribute to the broader understanding of the material properties of biomolecular condensates. The findings will have implications for unraveling the roles of these condensates in cellular organization and function, as well as their involvement in disease pathologies related to aberrant phase separation processes.


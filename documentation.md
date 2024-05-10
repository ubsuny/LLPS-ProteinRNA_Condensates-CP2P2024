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

# TensorFlow in MSD Calculation

TensorFlow is engineered for high-performance numerical computing, adept at managing multidimensional arrays and complex operations, which makes it invaluable for large-scale machine learning tasks. In our MSD calculation code, we utilized TensorFlow to orchestrate these computations within a graph-based architecture, an environment where operations are nodes and data are edges, expecting to leverage its potential for rapid, parallel processing: TensorFlow has ability to do many tasks at once to make our calculations fast.  However, in this scenario, the computational load did not necessitate such an elaborate setup. The overhead of initializing TensorFlow's environment overshadowed its benefits, as the task's simplicity didn't harness the full scope of TensorFlow's capabilities. Contrarily, NumPy, with its straightforward array operations and CPU-optimized performance, emerged as the more efficient tool for this specific MSD computation, delivering fast and precise results without the complexity of TensorFlow's computational graph. Moreover, we compared the performance and applicability of two prominent Python libraries, NumPy and TensorFlow, for computing MSD. Below is the code for Tensorflow:
```python
import tensorflow as tf
# Convert xpos and ypos to TensorFlow tensors
xpos_tf = tf.constant(xpos, dtype=tf.float32)
ypos_tf = tf.constant(ypos, dtype=tf.float32)

# Function to calculate MSD using TensorFlow
def calculate_msd_tensorflow(xpos, ypos, frametime, minframe):
    nData = tf.size(xpos)
    numberOfdeltaT = tf.cast(nData / 2, tf.int32)
    msd = tf.TensorArray(dtype=tf.float32, size=numberOfdeltaT, dynamic_size=False)

    # Function to compute the squared displacement
    def compute_displacement(delta_t, msd):
        deltax = xpos[1 + delta_t:nData - 1] - xpos[1:nData - 1 - delta_t]
        deltay = ypos[1 + delta_t:nData - 1] - ypos[1:nData - 1 - delta_t]
        squared_displacement = tf.square(deltax) + tf.square(deltay)
        mean_squared_displacement = tf.reduce_mean(squared_displacement)
        msd = msd.write(delta_t-1, (frametime * tf.cast(delta_t, tf.float32), mean_squared_displacement))
        return delta_t + 1, msd

    _, msd_final = tf.while_loop(
        cond=lambda delta_t, _: delta_t <= numberOfdeltaT,
        body=compute_displacement,
        loop_vars=(tf.constant(1), msd)
    )

    msd_result = msd_final.stack()
    return msd_result

# Call the function
msd_tf = calculate_msd_tensorflow(xpos_tf, ypos_tf, frametime, minframe)
msd_tf = msd_tf.numpy()  # Convert the result back to a NumPy array for further processing or analysis

```
In the `calculate_msd_tensorflow` function, we leverage several TensorFlow components: `tf.constant` is used to transform our position data into a stable form that TensorFlow can work with, forming the building blocks of the computation. `tf.TensorArray` acts as a dynamic list to store our results as we loop through calculations. The `tf.while_loop` construct efficiently repeats a block of calculations (squared displacements in our case) until a specified condition is no longer met, helping to manage memory more effectively and speed up the process on powerful hardware. `tf.square` performs an element-wise operation to square the difference between positions, which is a crucial step in MSD calculation. Finally, `tf.reduce_mean` is applied to average these squared differences, giving us the Mean Squared Displacement. These TensorFlow tools are designed to handle repetitive calculations and large arrays efficiently, potentially taking advantage of GPU or TPU acceleration to speed up the computation significantly.

## MSD Plot Explanation and Execution Time Comparison

![msd_plot](https://github.com/ubsuny/LLPS-ProteinRNA_Condensates-CP2P2024/assets/143649367/478e9229-fa3e-4f14-bfb2-2f50c4177513)


The provided MSD plot presents the time-dependent trajectory of the MSD values calculated using both NumPy and TensorFlow.

### Plot Analysis:
- The blue line with circle markers represents the NumPy results, showing a consistent and expected increase in MSD over time, which is typical for diffusive processes.
- The orange dashed line with cross markers illustrates the TensorFlow results, closely tracking the NumPy computation, suggesting numerical consistency across both methods.

### Execution Time Analysis:
- **NumPy Execution Time**: Rapid computation (`0.0113 seconds`) due to its highly optimized CPU array operations.
- **TensorFlow Execution Time**: Slower computation (`1.8179 seconds`) due to the overhead of initializing a computational graph, which is not offset by the complexity of the task. TensorFlow's design incorporates the building of a complex graph structure to map out all the computations before any actual number crunching occurs. This characteristic, while beneficial for computationally intensive tasks that can leverage parallelism, adds an unnecessary layer of complexity for straightforward calculations like MSD, thus prolonging the computation time without yielding additional benefits.  

The side-by-side MSD trajectory analysis emphasizes the critical aspect of tool selection in computational tasks. It underscores that the choice of computational framework should align with the complexity and nature of the task at hand. For calculations similar to MSD that are not inherently parallelizable or data-intensive, the direct computational abilities of NumPy are more advantageous, providing quick and accurate results without the need for the advanced setup that TensorFlow necessitates.  

For the MSD calculation examined here, TensorFlow is **not considered justifiable**. NumPy is more than sufficient for the task, offering faster execution and negligible differences in the accuracy of results. The advanced capabilities of TensorFlow, such as GPU acceleration and handling of massive datasets, do not come into play for this particular application. By strategically choosing between NumPy and TensorFlow based on the nature of the computational task, developers can ensure efficient resource utilization and optimal performance.

## Viscosity Estimation for dT40 and [RGRGG]5 Condensates

### dT40-[RGRGG]5 Condensate Results

#### Experimental Setup and Results

The dT40 and [RGRGG]5 were mixed to form condensates in a buffer solution designed to mimic cellular conditions. Fluorescently labeled polystyrene beads (200 nm) were incorporated into the mixture to track and measure movement within the condensates, utilizing Video Particle Tracking (VPT).

#### Materials and Methods

- **Buffer Components**: 
  - **25 mM Tris-HCl**: Stabilizes pH.
  - **25 mM NaCl**: Provides ionic strength to stabilize charges on biomolecules.
  - **20 mM DTT**: Prevents the formation of disulfide bonds between cysteine residues.

The condensate sample was sandwiched between a tween 20-coated coverslip and a slide with mineral oil to prevent evaporation and maintain sample integrity. High-resolution imaging was performed using an epifluorescence microscope with a 100x oil-immersion objective lens.  

### Equations Used for Analysis

**Mean Squared Displacement (MSD) Equation**:  

$MSD(\tau) = 4D\tau^\alpha$  

  Where:
  - $D$ is the diffusion coefficient, which measures how fast particles spread out over time.
  - $\tau$ is the lag time over which the displacement of the particles is measured.
  - $\alpha$ is the diffusive exponent—classically 1 for Brownian motion in viscous fluids, indicating a linear relationship between MSD and time, characteristic of normal diffusion. Values different from 1 suggest anomalous diffusion: $\alpha < 1$ indicates subdiffusive (slow diffusion), and $\alpha > 1$ indicates superdiffusive (fast diffusion).  

  This equation is fundamental for analyzing the random motion of particles within the condensates. It quantifies how much a particle's position changes over time, providing insights into the dynamics within the medium. The diffusion coefficient \( D \) measures the rate of particle dispersion, while \( \alpha \) indicates the nature of the diffusion (normal, subdiffusive, superdiffusive).

**Stokes-Einstein Equation**:
  
  $\eta = \frac{k_BT}{6\pi rD}$
  
  Where:
  - $\eta$ is viscosity, representing the fluid's resistance to gradual deformation by shear stress or tensile stress.
  - $k_B$ is the Boltzmann constant, which links temperature to energy.
  - $T$ is the absolute temperature of the fluid.
  - $r$ is the radius of the bead, serving as a scale of the particle size in the fluid.
  - $D$ is the diffusion coefficient from the MSD equation, providing a measure of how quickly particles are moving through the fluid.  

  This equation is used to calculate the viscosity of the fluid based on the diffusion coefficient derived from the MSD data. Viscosity is a key parameter in understanding how biomolecular condensates influence cellular processes. It measures the fluid's internal resistance to flow and can indicate how substances move through the condensate, affecting reaction rates and molecular interactions.

These equations are integral to the analysis as they allow for the quantification of physical properties that are critical to understanding the behavior of biomolecular condensates under various conditions. By measuring how particles move in response to their environment, researchers can infer the mechanical and dynamic properties of the condensates.


#### Data Analysis

Using the TrackMate plugin and custom Python scripts, the trajectories of beads within the condensates were tracked. The Mean Squared Displacement (MSD) for each bead was calculated, reflecting their motion in the condensates.

The ensemble-averaged MSD was derived from individual trajectories, representing overall particle dynamics.
![Tracksindividual_track](https://github.com/ubsuny/LLPS-ProteinRNA_Condensates-CP2P2024/assets/143649367/822baced-b31d-4a30-85fa-74e18a9f9a37)    
*Figure: Random Walk Trajectory of a particle within the condensate.*  

---------------------------------------------------------------------------------------------------------------------------------------------------  

![Screenshot from 2024-05-10 12-21-38](https://github.com/ubsuny/LLPS-ProteinRNA_Condensates-CP2P2024/assets/143649367/4e9969be-76f1-4072-9b6f-8b5fc50df282)
*Figure: MSD for multiple particles over time, showing a linear trend indicative of diffusive behavior.*    

---------------------------------------------------------------------------------------------------------------------------------------------------
![TracksMSDfit](https://github.com/ubsuny/LLPS-ProteinRNA_Condensates-CP2P2024/assets/143649367/7fef920a-7d7e-44d9-853b-cff887ee8f02)  
*Figure: Fitting of the MSD curve, used to calculate the diffusion coefficient and viscosity.*  

---------------------------------------------------------------------------------------------------------------------------------------------------
#### Results

- **Diffusion Coefficient**: $\ D = 1.9342 \times 10^{-4} \, \text{µm}^2/\text{s} \$
- **Viscosity**: $\ \eta = 2.22 \, \text{Pa}\cdot\text{s} \$, indicating a relatively viscous medium compared to water $\ \eta_{\text{water}} = 0.001 \, \text{Pa}\cdot\text{s} \$.
- **Alpha Value**: $\ \alpha = 1.4237 \$, suggesting that the particle motion deviates from classical Brownian motion and reflects a super-diffusive process.

#### Conclusion

The experimental results successfully quantified the diffusive properties of particles within the dT40-[RGRGG]5 condensates. The viscosity derived from the MSD fitting provides crucial insights into the mechanical resistance of the fluid, with potential implications for material design and drug delivery systems.  

The project on viscosity estimation of DNA-protein condensates using Video Particle Tracking (VPT) introduces novel approaches in biophysics research. It leverages high-resolution epifluorescence microscopy and VPT to precisely track particle dynamics within condensates, offering advanced insights into the rheological properties of biomolecular condensates. This innovative use of VPT in a simulated cellular environment not only enhances the understanding of condensate behavior under various conditions but also provides a detailed characterization of their mechanical properties, thus contributing valuable new insights into cellular organization and dynamics.




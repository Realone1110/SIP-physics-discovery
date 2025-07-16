# SIP Physics Discovery

## License
Released under the MIT License (see `LICENSE` for full text).

## Code for Consistent Discovery of Dynamical Systems Using Stochastic Inverse Modeling

This repository contains the raw Python code and Jupyter notebooks for the Physics Discovery Paper Project, focusing on the consistent discovery of dynamical systems through stochastic inverse modeling techniques.

## 📁 Repository Structure

### Case Studies
Each case study is organized in its own dedicated folder with complete implementations:

- **`All infiltration systems/`** - Comprehensive infiltration system analyses
- **`Hudson Bay/`** - Hudson Bay case study implementation  
- **`Lorrentz Sim/`** - Lorenz system simulations and analysis
- **`Lotka-Volterra simulation/`** - Predator-prey model implementations

### Supporting Files
- **`Scripts/`** - Helper functions and utility scripts used across case studies
- **`Embedded Images/obs_data/`** - Generated visualizations that are embedded within the submitted manuscript 
- **Individual Notebooks** - Standalone analysis files:
  - `class_sip_learning.ipynb` - Main SIP learning class implementation
  - `Untitled.ipynb` & `Untitled1.ipynb` - Exploratory analysis notebooks (Please ignore this)

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required packages (install via pip):
  ```bash
  pip install numpy scipy matplotlib pandas jupyter
  ```

### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Realone1110/SIP-physics-discovery.git
   cd SIP-physics-discovery
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Navigate to any case study folder and open the corresponding `.ipynb` file

## 📊 Case Study Descriptions

### All Infiltration Systems
Contains comprehensive analysis of various infiltration physics models, including:
- High viscous SIP implementations
- Infiltration physics simulations
- 

### Hudson Bay
Complete analysis and modeling specific to Hudson Bay dynamics.

### Lorenz System
Classic chaotic system analysis including:
- Attractor visualization
- Model parameter physics recovery

### Lotka-Volterra System
Predator-prey population dynamics modeling with:
- Parameter estimation for single path Lotka Volterra model
- Parameter estimation for multi-path path Lotka Volterra model

## 🔧 Helper Functions

Helper functions are provided in two locations:
1. **Within individual notebooks** - Case-specific utility functions
2. **`Scripts/` folder** - Shared functions used across multiple case studies

## 📈 Methodology

This project implements **Stochastic Inverse Modeling for Physics Discovery (SIP)** for physics discovery, focusing on:
- Consistent parameter estimation
- Uncertainty quantification
- Robust model selection
- Dynamical system identification

## 🔬 Research Context

This code supports research in:
- Inverse problem solving in physics
- Stochastic modeling techniques
- Dynamical systems recovery
- 

## 📝 File Formats

- **`.ipynb`** - Jupyter notebooks with complete analysis workflows
- **`.py`** - Python scripts with reusable functions
- **Data files** - Observational and simulation data

## 🤝 Contributing

This repository contains research code for academic purposes. For questions or collaboration inquiries, please open an issue or contact olabiyiridwan12@gmail.com.

## 📚 Citation

If you use this code in your research, please cite the associated Physics Discovery Paper (citation details to be updated upon publication).

## 📧 Contact

For questions about the implementation or methodology, please contact olabiyiridwan12@gmail.com.

**Note**: This repository contains research code that may require domain-specific knowledge in physics and dynamical systems theory for full utilization.

# Toy Jets Shower

### ****

Note that this is an early development version. 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## Introduction



**Relevant Structure**:

- [`data`](data/): Dir with the trees.

- [`analysis`](analysis): 

    -[`likelihood.py`](showerSim/likelihood.py): Calculate the log likelihood of a splitting node and of (a branch of) a tree. There are examples on how to run it at the end of the script.

- [`showerSim`](showerSim/): Dir with the simulation code.

    -[`exp2DShowerTree.py`](showerSim/exp2DShowerTree.py): Parton shower code to generate the trees. 


-[`run2DShower.py`](showerSim/run2DShower.py): Run the parton shower code in [`showerSim`](showerSim/).
    
- [`visualized-recursion_2D.ipynb`](visualized-recursion_2D.ipynb): Jet trees visualization.



##### **Running the simulation locally as a python package:**


1. Clone the ToyJetsShower repository
2. `cd `[`ToyJetsShower`](.)
3. `pip install -e .`
4.`>>> import run2DShower`













Running the simulation as a python package:

- cd to main dir
- pip install -e .









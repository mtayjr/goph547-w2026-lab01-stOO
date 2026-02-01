# GOPH 547
*Semester:* W2026
*Instructor:* B. Karchewski
*Author(s):* O. Ojo


# Lab 00
This repository contains my solution for **Lab 00** in GEOPH 547.

The goal of this lab was to get comfortable with python package structure, Github creation and version control, Numpy array operations, 
and some basic image processing using Numpy and Matplotlib.

# Github Repository Overview
The repository is organized using a standard Python 'src/' layout:
geoph547-w2026-lab00-stOO
 src/
  ---goph547lab00/
         ---------init.py
         ---------arrays.py
 examples/
         ---------driver.py
         ---------rock_canyon.jpg
         ---------rock_canyon_RGB_summary.png
 README.md
 pyproject.toml
 .gitignore


All reuseable source code lives in 'src/goph547labOO/'
Example runs and scripts are in 'examples/'
The image used in Part B and the final saved plot are also stored in 'examples/'

#Environment and Installation

A Python virtual environment was created to keep dependencies isolated:

'''bash
python3 -m venv .venv
source .venv/bin/activate

The project dependencies are defined in pyproject.toml.
The package was installed in editable mode using "pip install -e" this allows changes made in the 
src/ directory to be picked up immediately when running the example scripts.

#GitHub Setup

A github repository was created for this lab and connected locally using SSH. All work was committed 
incrementally and pushed to GitHub as the lab progressed. The repository follows a clean structure so
that code, examples, and outputs are easy to locate and run.

#Running the Code

All parts of the lab can be run from the project root using "python examples/driver.py"
This script runs both Part A and Part B of the assignment.

#Part A - Package Test

Part A checks that the Python package is setup correctly. 
The function square_ones(n) is defiend in arrays.py.
It returns as n x n Numpy array of ones.
In driver.py, the output of square_ones(3) is compared against numpy.ones((3,3)) to 
confirm the function works as expected.
This confirms that the package can be imported and used correctly.

#Part Bi - Numpy Array Operations

The first section of Part B focuses on basic numoy operations:
1. Creating a (3 x 5) array of ones.
2. Creating a (6 x 3) array filled with NaN values.
3. Creating a column vector of odd numbers between 44 and 75.
4. Computing the sum of that vector.
5. Defining a specific array A.
6. Creating array B using a single numpy command.
7. Performing element-wise multiplication of A and B
8. Computing the dot product of A and B
9. Computing the cross product of A and B.
All results are printed to the terminal for inspection.

#Part Bii - Image Processing 
The second section of part B works with image data:
10. The image rock_canyon.jpg is loaded into a numpy array.
11. The image is displayed and its array shape is reported.
12. The image is converted to grayscale and the new array shape is reported.
13. A smaller grayscale sub-image focusing on the rock pillar is extracted.
14. Mean RGB values are computed along the x-direction.
15. Mean RGB values are computed along the y-direction.
16. These results are plotted as two subplots and saved as 
rock_canyon_RGB_summary.png using savefig()

Only the final RGB summary figure is saved to disk, as required by the lab. 
The intermediate images are displayed for visualization only. 

  
 

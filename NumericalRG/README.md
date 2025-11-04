# Numerical Renormalization Group (NRG) for the Single Impurity Anderson Model (SIAM)

This repository provides a numerical implementation of the **Numerical Renormalization Group (NRG)** method to accurately solve the quantum many-body problem described by the **Single Impurity Anderson Model (SIAM)**.

---

## Overview

The **Single Impurity Anderson Model (SIAM)** is a foundational model in condensed matter physics used to describe phenomena like the **Kondo effect**, where a single magnetic impurity interacts with a non-interacting metallic host.

The **Numerical Renormalization Group (NRG)** is a powerful and exact numerical method, pioneered by Kenneth G. Wilson, specifically designed to handle the multi-scale physics inherent in quantum impurity problems. 
It achieves this by iteratively diagonalizing the Hamiltonian on a logarithmic energy grid, which efficiently captures the low-energy fixed points and thermodynamic properties of the system.

This repository provides:
* A **Python implementation** of the NRG algorithm based on tensor network.
* The necessary tools to construct the SIAM Hamiltonian.
* A guided tutorial demonstrating how to calculate key physical quantities, such as the **Energy level flow diagram**

---

## Installation

To use this code, you'll need Python and a few common scientific computing libraries.

### Prerequisites

* **Python 3.x**
* **NumPy** (for numerical operations)
* **SciPy** (for sparse matrices and linear algebra)
* **Matplotlib** (for plotting)
* **Jupyter** (to run the tutorial notebook)

### Setup Instructions

1.  **Clone the repository:**
2.  **Install the required packages:**

---

## Getting Started and Tutorial

### The Tutorial Notebook: `SIAM_NRG.ipynb`
Please review the tutorial file and report any errors you find.

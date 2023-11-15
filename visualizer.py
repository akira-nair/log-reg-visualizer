#!/usr/bin/env python
'''
File        :   visualizer.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Plots a decision boundary for a
                simple half-space classifier trained
                on two features.
'''

"""
USAGE:
python visualizer.py
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def prompt_args():
    """
    Prompts user for arguments
    """
    print(f"""\nWelcome to the binary classifier visualizer!\nPlease provide the """
    """bounds of the axes and the weights of the halfspace classifier.\n=====\n""")
    # prompt for axes bounds
    x1_min = float(input('x1 min: '))
    x1_max = float(input('x1 max: '))
    x2_min = float(input('x2 min: '))
    x2_max = float(input('x2 max: '))

    weight1 = float(input('weight 1: '))
    weight2 = float(input('weight 2: '))
    bias = float(input('bias: '))
    print(f"\nEquation given:\nh(x) = sigmoid({weight1}x_1 + {weight2}x_2 + {bias})")
    weights = np.array([weight1, weight2, bias])
    # convert to namespace
    args = argparse.Namespace(x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max, weights=weights)
    return args

def plot_binary_probability_space(binary_model: argparse.Namespace, resolution: int = 50):
    """
    Given a set of parameters, plot a heatmap of the probability outputs given by the weights

    Args:
        binary_model (argparse.Namespace): _description_
    """
    # create meshgrid
    x1 = np.linspace(binary_model.x1_min, binary_model.x1_max, resolution)
    x2 = np.linspace(binary_model.x2_min, binary_model.x2_max, resolution)
    X1, X2 = np.meshgrid(x1, x2)

    # calculate probability for each point using a sigmoid activation activation
    Z = 1 / (1 + np.exp(-1 * (binary_model.weights[0] * X1 + binary_model.weights[1] * X2 + binary_model.weights[2])))

    # plot heatmap
    plt.pcolormesh(X1, X2, Z, cmap='cool', vmin=0, vmax=1, shading='auto')
    plt.colorbar()

    # plot line of separation
    plt.contour(X1, X2, Z, levels=[0.5], colors='white')
    model_eq = f"{binary_model.weights[0]}x_1 + {binary_model.weights[1]}x_2 + {binary_model.weights[2]}"
    plt.title(f"Probability Space of Binary Classifier\n{model_eq}")
    plt.show()

def main():
    # prompt for arguments
    args = prompt_args()
    # plot probability space
    plot_binary_probability_space(args)

if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
import numpy as np
import os
# Sample DNA sequence
dna = np.load("models/dna/numpy_dna.npy")
print("A&C&T&G"*7)
print("l "*28)
for element in dna:
    dna_int = element.tolist()
    dna_str = [str(i) for i in dna_int]
    print("&".join(dna_str[0:32]))
    
# Plot the DNA s

import numpy as np
import streamlit as st
from pyDecision.algorithm import promethee_iv
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')

# PROMETHEE IV Interface

def main():
    st.title("PROMETHEE IV Streamlit Interface")
    
    W = st.text_area("Weights (W) in preference criterion:", "9.00, 8.24, 5.98, 8.48")
    P = st.text_area("Indifference thresholds (P):", "0.5, 0.5, 0.5, 0.5")
    Q = st.text_area("Preference thresholds (Q):", "0.3, 0.3, 0.3, 0.3")
    S = st.text_area("Preference function sharpness (S):", "0.4, 0.4, 0.4, 0.4")
    F = st.text_area("Preference functions ('t1' to 't7'):", "t5, t5, t5, t5")
    dataset_input = st.text_area("Dataset (each row on a new line, values separated by commas):", "8.840, 8.790, 6.430, 6.950\n8.570, 8.510, 5.470, 6.910\n7.760, 7.750, 5.340, 8.760\n7.970, 9.120, 5.930, 8.090\n9.030, 8.970, 8.190, 8.100\n7.410, 7.870, 6.770, 7.230")
    
    W = list(map(float, W.split(',')))
    P = list(map(float, P.split(',')))
    Q = list(map(float, Q.split(',')))
    S = list(map(float, S.split(',')))
    F = F.split(',')
    dataset_rows = dataset_input.strip().split("\n")
    dataset = np.array([list(map(float, row.strip().split(','))) for row in dataset_rows])
    
    if st.button("Compute"):
        try:
            p4 = promethee_iv(dataset, W = W, Q = Q, S = S, P = P, F = F, sort = True, steps = 0.001, topn = 10, graph = True, verbose = True)
            st.write("Net Flows: ", p4)
        except Exception as e:
            st.write("An error occurred: ", str(e))

if __name__ == '__main__':
    main()

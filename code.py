import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import streamlit as st

###############################################################################

# Function: Distance Matrix
def distance_matrix(dataset, criteria = 0):
    distance_array = np.zeros(shape = (dataset.shape[0],dataset.shape[0]))
    for i in range(0, distance_array.shape[0]):
        for j in range(0, distance_array.shape[1]):
            distance_array[i,j] = dataset[i, criteria] - dataset[j, criteria] 
    return distance_array

# Function: Preferences
def preference_degree(dataset, W, Q, S, P, F):
    pd_array = np.zeros(shape = (dataset.shape[0],dataset.shape[0]))
    for k in range(0, dataset.shape[1]):
        distance_array = distance_matrix(dataset, criteria = k)
        for i in range(0, distance_array.shape[0]):
            for j in range(0, distance_array.shape[1]):
                if (i != j):
                    if (F[k] == 't1'):
                        if (distance_array[i,j] <= 0):
                            distance_array[i,j]  = 0
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't2'):
                        if (distance_array[i,j] <= Q[k]):
                            distance_array[i,j]  = 0
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't3'):
                        if (distance_array[i,j] <= 0):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > 0 and distance_array[i,j] <= P[k]):
                            distance_array[i,j]  = distance_array[i,j]/P[k]
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't4'):
                        if (distance_array[i,j] <= Q[k]):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > Q[k] and distance_array[i,j] <= P[k]):
                            distance_array[i,j]  = 0.5
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't5'):
                        if (distance_array[i,j] <= Q[k]):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > Q[k] and distance_array[i,j] <= P[k]):
                            distance_array[i,j]  =  (distance_array[i,j] - Q[k])/(P[k] -  Q[k])
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't6'):
                        if (distance_array[i,j] <= 0):
                            distance_array[i,j]  = 0
                        else:
                            distance_array[i,j] = 1 - math.exp(-(distance_array[i,j]**2)/(2*S[k]**2))
                    if (F[k] == 't7'):
                        if (distance_array[i,j] == 0):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > 0 and distance_array[i,j] <= S[k]):
                            distance_array[i,j]  =  (distance_array[i,j]/S[k])**0.5
                        elif (distance_array[i,j] > S[k] ):
                            distance_array[i,j] = 1
        pd_array = pd_array + W[k]*distance_array
    pd_array = pd_array/sum(W)
    return pd_array

# Function: Rectangular Integration
def integration_prefence_degree(dataset, W, Q, S, P, F, steps = 0.01):
    pd_array = np.zeros(shape = (dataset.shape[0],dataset.shape[0]))
    for k in range(0, dataset.shape[1]):
        distance_array = distance_matrix(dataset, criteria = k)
        for i in range(0, distance_array.shape[0]):
            for j in range(0, distance_array.shape[1]):
                if (i != j):
                    area      = 0
                    f         = 0
                    distance  = steps/2
                    while (distance <= distance_array[i,j] and  distance_array[i,j] > 0):
                        if (F[k] == 't1'):
                            if (distance <= 0):
                                f = 0
                            else:
                                f = 1
                        if (F[k] == 't2'):
                            if (distance <= Q[k]):
                                f  = 0
                            else:
                                f = 1
                        if (F[k] == 't3'):
                            if (distance <= 0):
                                f = 0
                            elif (distance > 0 and distance <= P[k]):
                                f = distance/P[k]
                            else:
                                f = 1
                        if (F[k] == 't4'):
                            if (distance <= Q[k]):
                               f = 0
                            elif (distance > Q[k] and distance <= P[k]):
                                f = 0.5
                            else:
                                f = 1
                        if (F[k] == 't5'):
                            if (distance <= Q[k]):
                                f = 0
                            elif (distance > Q[k] and distance <= P[k]):
                                f =  (distance - Q[k])/(P[k] -  Q[k])
                            else:
                               f = 1
                        if (F[k] == 't6'):
                            if (distance <= 0):
                                f = 0
                            else:
                                f = 1 - math.exp(-(distance**2)/(2*S[k]**2))
                        area = area + f*steps
                        distance = distance + steps
                    distance_array[i,j] = area
        pd_array = pd_array + W[k]*distance_array
    pd_array = pd_array/sum(W) 
    return pd_array

# Function: Rank 
def ranking(flow):    
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i           
    for i in range(0, rank_xy.shape[0]):
        if (flow[i,1] >= 0):
            plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.5, 0.8, 1.0),))
        else:
            plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12,ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.8, 0.8),))            
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    xmin = np.amin(rank_xy[:,0])
    xmax = np.amax(rank_xy[:,0])
    axes.set_xlim([xmin-1, xmax+1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show() 
    return

###############################################################################

# Function: Promethee IV
def promethee_iv(dataset, W, Q, S, P, F, sort = True, steps = 0.001, topn = 0, graph = False, verbose = True):
    pd_matrix  = integration_prefence_degree(dataset, W, Q, S, P, F, steps)
    flow_plus  = np.sum(pd_matrix, axis = 1)/(pd_matrix.shape[0] - 1)
    flow_minus = np.sum(pd_matrix, axis = 0)/(pd_matrix.shape[0] - 1)
    flow       = flow_plus - flow_minus 
    flow       = np.reshape(flow, (pd_matrix.shape[0], 1))
    flow       = np.insert(flow, 0, list(range(1, pd_matrix.shape[0]+1)), axis = 1)
    if (sort == True or graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
    if (topn > 0):
        if (topn > pd_matrix.shape[0]):
            topn = pd_matrix.shape[0]
        if (verbose == True):
            for i in range(0, topn):
                print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3)))  
    if (graph == True):
        ranking(flow)
    return flow

def promethee_iv_interface(W, P,S, Q, F, dataset):
    # Convertir les entrées en listes
    W = list(map(float, W.split(',')))
    P = list(map(float, P.split(',')))
    Q = list(map(float, Q.split(',')))
    F = list(map(str, F.split(',')))
    S= list(map(float, S.split(',')))

    # Appeler la fonction PROMETHEE IV
    result = promethee_iv(dataset, W, Q, S, P, F, sort=True, steps=0.001, topn=15, graph=True, verbose=True)

    return result

# Fonction principale de Streamlit
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Promethee IV - Interface")

    # Ajouter des champs pour les paramètres
    W = st.text_input("Les poids W (séparés par une virgule):", "9.00, 8.24, 5.98, 8.48")
    P = st.text_input("Valeur de P (séparée par une virgule):", "0.5,  0.5,  0.5,  0.5")
    Q = st.text_input("Valeur de Q (séparée par une virgule):", "0.3,  0.3,  0.3,  0.3")
    F = st.text_input("Valeur de fonction de préférence (séparée par une virgule):", "t4, t4, t4, t4")
    S = st.text_input("Valeur de segma(séparée par une virgule):", "0.4,  0.4,  0.4,  0.4")


    # Ajouter un champ pour la matrice des données
    dataset_input = st.text_area("Matrice des données (séparée par des espaces, lignes séparées par des sauts de ligne):")
    #8.840, 8.790, 6.430, 6.950
    #8.570, 8.510, 5.470, 6.910
    #7.760, 7.750, 5.340, 8.760
    #7.970, 9.120, 5.930, 8.090
    #9.030, 8.970, 8.190, 8.100
    #7.410, 7.870, 6.770, 7.230
    dataset_rows = dataset_input.strip().split("\n")
    dataset = [list(map(float, row.strip().split())) for row in dataset_rows]

    # Bouton pour calculer et afficher les résultats
    if st.button("Calculer et Afficher"):
        result = promethee_iv_interface(W, P, Q, S, F, np.array(dataset))

        # Diviser l'espace en colonnes
        col1, col2 = st.columns(2)

        # Afficher le tableau des valeurs dans la première colonne
        col1.write("Classement:")
        col1.write(result)

        # Afficher le graphique de surclassement dans la deuxième colonne
        col2.write("Graphique de surclassement:")
        ranking(result)
        col2.pyplot()
        

if __name__ == '__main__':
    main()

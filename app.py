import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

# --- Geometry functions ---
def midpoint(listT):
    listmid = []
    for T in listT:
        listmid.extend([
            (T[1] + T[2]) / 2,
            (T[2] + T[0]) / 2,
            (T[0] + T[1]) / 2
        ])
    return np.array(listmid)

def sym(P, C):
    return 2. * C - P

# --- Initialize session state ---
if "fig" not in st.session_state:
    st.session_state.fig, st.session_state.ax = plt.subplots()
    st.session_state.frame_counter = 0
    st.session_state.iteration_data = []
    st.session_state.all_midpoints = []
    st.session_state.scatter = None
    st.session_state.initialized = False

# --- Reset logic ---
def reset():
    st.session_state.fig, st.session_state.ax = plt.subplots()
    ax = st.session_state.ax
    ax.axis("off")
    st.session_state.frame_counter = 0
    st.session_state.iteration_data = []
    st.session_state.all_midpoints = []

    # Random triangle
    A, B, C = np.random.rand(2), np.random.rand(2), np.random.rand(2)
    T0 = np.array([A, B, C, A])
    center = (A + B + C) / 3
    ax.set_xlim(-1.7 + center[0], 1.7 + center[0])
    ax.set_ylim(-1.7 + center[1], 1.7 + center[1])
    ax.plot(T0[:, 0], T0[:, 1], color="black")

    # Compute iterations
    N = 22
    listT = [T0[0:3]]
    listallT = [T0[0:3]]
    cmap = plt.cm.get_cmap("viridis", N)  # Create a colormap with N discrete colors

    for step in range(N):
        listmid = midpoint(listT)
        new_triangles = []
        listTnew = []
        color = cmap(step)  # Get a color based on the current iteration

        for k, T in enumerate(listT):
            for i in range(3):
                Anew = T[(i + 1) % 3]
                Bnew = T[(i + 2) % 3]
                Pnew = sym(T[i], listmid[3 * k + i])
                Tnew = np.array([Anew, Pnew, Bnew])

                listTarray = np.array(listallT)
                listerrs = [
                    np.sum([norm(Tnew[i] - listTarray[:, j], axis=1) for i, j in enumerate(order)], axis=0)
                    for order in [[0, 1, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0], [1, 0, 2], [0, 2, 1]]
                ]
                min_err = np.min([np.min(e) for e in listerrs])
                if min_err > 1e-12:
                    new_triangles.append((Tnew, color))  # Store triangle *with color*
                    listTnew.append(Tnew)
                    listallT.append(Tnew)

        st.session_state.iteration_data.append((listmid, new_triangles))
        listT = listTnew.copy()

    st.session_state.scatter = ax.scatter([], [], color="black", s=6, alpha=1.0, zorder=3)
    st.session_state.initialized = True

# --- Step function ---
def step():
    if not st.session_state.initialized:
        return

    ax = st.session_state.ax
    i = st.session_state.frame_counter // 2
    if i >= len(st.session_state.iteration_data):
        return

    listmid, new_triangles = st.session_state.iteration_data[i]

    if st.session_state.frame_counter % 2 == 0:
        # Add midpoints
        st.session_state.all_midpoints.extend(listmid)
        st.session_state.scatter.set_offsets(np.array(st.session_state.all_midpoints))
    else:
        # Draw new triangles
        for Tnew, color in new_triangles:
            ax.plot(Tnew[[0, 1, 2, 0], 0], Tnew[[0, 1, 2, 0], 1], color=color)


    st.session_state.frame_counter += 1

# --- GUI ---
st.title("Random triangle tessellation")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Random triangle"):
        reset()
with col2:
    if st.button("Step"):
        step()

# Only display the figure once
if st.session_state.initialized:
    st.pyplot(st.session_state.fig)

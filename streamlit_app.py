# %%

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# %% sliders

slider_steps_params = [
    "simulation steps",  # label
    0,  # min
    100,  # max
    50,  # start value
    1,  # step
]


slider_initial_params = [
    "initial state: prob(think)",  # label
    0.0,  # min
    1.0,  # max
    0.5,  # start value
    0.001,  # step
    "%f",  # format
]


slider_pt2t_params = [
    "prob: think to think",  # label
    0.0,  # min
    1.0,  # max
    0.6,  # start value
    0.001,  # step
    "%f",  # format
]

slider_pnt2t_params = [
    "prob: no think to think",  # label
    0.0,  # min
    1.0,  # max
    0.05,  # start value
    0.001,  # step
    "%f",  # format
]


steps = st.sidebar.slider(*slider_steps_params)
slider_steps_params[3] = steps

initial_state = st.sidebar.slider(*slider_initial_params)
slider_initial_params[3] = initial_state

pt2t = st.sidebar.slider(*slider_pt2t_params)
slider_pt2t_params[3] = pt2t

pnt2t = st.sidebar.slider(*slider_pnt2t_params)
slider_pnt2t_params[3] = pnt2t

# st.sidebar.markdown("Transition probabilities")
# st.sidebar.write("$P(think|think)$ = ", np.round(pt2t, 3))
# st.sidebar.write("$P(nothink|think)$ = ", np.round(1 - pt2t, 3))

# st.sidebar.write("$P(think|nothink)$ = ", np.round(pnt2t, 3))
# st.sidebar.write("$P(nothink|nothink)$ = ", np.round(1 - pnt2t, 3))

# st.sidebar.write("initial $P(think)$ = ", np.round(initial_state, 3))
# st.sidebar.write("initial $P(nothink)$ = ", np.round(1 - initial_state, 3))


# %%

M = np.array([pt2t, pnt2t, 1 - pt2t, 1 - pnt2t]).reshape(2, -1)
# M = np.array([0.8, 0.25, 0.2, 0.75]).reshape(2, -1)
M_temp = M.copy()
state = np.array([initial_state, 1 - initial_state]).reshape(2, -1)
# state = np.array([1.0, 0.0]).reshape(2, -1)

# steps = 100
probs = np.zeros((2, steps + 1))
probs[:, 0] = state[:, 0]
# probs

for s in range(steps):
    if s > 0:
        M_temp = M @ M_temp
    outcome = M_temp @ state
    probs[:, s + 1] = outcome[:, 0]
    # print(f"outcome: \n{outcome}")

dt0 = pd.DataFrame(probs.reshape(-1, 1), columns=["prob"])
dt0["step"] = list(range(steps + 1)) * 2
dt0["state"] = ["think"] * (steps + 1) + ["nothink"] * (steps + 1)
# probs

fig = (
    alt.Chart(dt0)
    .mark_bar()
    .encode(
        x=alt.X("step", scale=alt.Scale(domain=[0, steps + 1])),
        y=alt.Y("sum(prob)", axis=alt.Axis(title="Sum of prob")),
        color="state",
        tooltip=["prob", "step", "state"],
    )
    .interactive()
)
# fig

# %%

st.markdown("# Markov model of accuracy nudge")
st.markdown("## Initial state and transition matrix")
col1, col2 = st.beta_columns([0.5, 0.5])

with col1:
    dt2 = pd.DataFrame(state, columns=["prob"])
    dt2[""] = ["think(t)", "nothink(t)"]
    dt2 = dt2.set_index([""]).reset_index()
    st.table(dt2)
with col2:
    dt1 = pd.DataFrame(M, columns=["think", "nothink"])
    dt1[""] = ["think(t+1)", "nothink(t+1)"]
    dt1 = dt1.set_index([""]).reset_index()
    st.table(dt1)


st.markdown(f"## Simulated results ({steps} steps)")
_, col_fig, _ = st.beta_columns([0.05, 0.8, 0.05])  # hack to center figure
with col_fig:
    st.altair_chart(fig, use_container_width=True)

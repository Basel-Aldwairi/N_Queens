# Imports
import streamlit as st
import NQ_functions
import time

# Page title
st.set_page_config(page_title='N-Queens Problem', page_icon='data/queen.png')

# session states for used variables
if 'algorithm' not in st.session_state:
    st.session_state.algorithm = 'BT'
if 'N' not in st.session_state:
    st.session_state.N = 8
if 'expanded_nodes' not in st.session_state:
    st.session_state.expanded_nodes = 0
if 'solved_board' not in st.session_state:
    st.session_state.solved_board = []
if 'changed_options' not in st.session_state:
    st.session_state.changed_options = False
if 'algorithm_time' not in st.session_state:
    st.session_state.algorithm_time = 0


# Title
st.markdown(f'''
# N-Queens Problem

#### Basel Al-Dwairi

___
''')


# Sidebar for options
st.sidebar.title('Options')

# Choose algorithm
algorithm = st.sidebar.radio('Algorithms : ', options=['Backtracking (BT)',
                                   'BT with Forward Checking',
                                   'Min Conflcts'])

# Update session state
prev_algorithm = st.session_state.algorithm
if algorithm == 'Backtracking (BT)':
    st.session_state.algorithm = 'BT'
elif algorithm == 'BT with Forward Checking':
    st.session_state.algorithm = 'BTFC'
elif algorithm == 'Min Conflcts':
    st.session_state.algorithm = 'MC'

# Check if options changes
if prev_algorithm != algorithm:
    st.session_state.changed_options = True

# Choose N
prev_N = st.session_state.N
N = st.sidebar.select_slider('N : ', range(1,33), value=8)
st.session_state.N = N

# Check if N changed
if prev_N != N:
    st.session_state.changed_options = True

# Algorithm map, from variable, to readable string
alogrithm_names = {
    'BT' : 'Backtracking (BT)',
    'BTFC' : 'BT with Forward Checking',
    'MC' : 'Min Conflcts'

}

# Show choosen algorithm and N
st.markdown(f'''
Algorithm : {alogrithm_names[st.session_state.algorithm]}

N :{st.session_state.N}
''')

# Pick the correct algorithm
algorithm_func = None
match st.session_state.algorithm:
    case 'BT':
        algorithm_func = NQ_functions.n_queens_backtracking
    case 'BTFC':
        algorithm_func = NQ_functions.n_queens_backtracking_with_forwardchecking
    case 'MC':
        algorithm_func = NQ_functions.n_queens_min_conflicts

# Run algorithm button
run_button = st.button('Run Algorithm')

# When run button is pressed
if run_button:
    if st.session_state.N in {2,3}:
        st.markdown(f'''
No Solutions for : {st.session_state.N}
''')
    else:
        st.session_state.changed_options = False


        # Run algorithm
        start_time = time.time()
        with st.spinner('Running Algorithm'):
            solved_board, expanded_nodes =  algorithm_func(N)
        algorithm_time = time.time() - start_time

        # Update session states
        st.session_state.solved_board = solved_board
        st.session_state.expanded_nodes = expanded_nodes
        st.session_state.solved_board = solved_board
        st.session_state.algorithm_time = algorithm_time

# If algorithm ran and no options were changed
if not st.session_state.changed_options:

    # Show preformance metrics
    st.markdown(f'''
    Nodes Expanded : {st.session_state.expanded_nodes}
    
    Time taken to solve : {st.session_state.algorithm_time:.5f}s
''')

    # Draw board
    board_img = NQ_functions.draw_board(st.session_state.solved_board)
    st.image(board_img)

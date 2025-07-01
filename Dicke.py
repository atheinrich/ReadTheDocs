########################################################################################################################################################
# Dicke model
########################################################################################################################################################

########################################################################################################################################################
# Summary
""" Command line example
    
    help(help)                                # view some tool descriptions
    sys = examples(7)                         # load a preset example
    sys.plot('occupation')                    # plot ⟨n⟩ and ⟨J_z⟩
    sys = System(1,1,40,4)                    # generate a new system with custom parameters
    sys.select('half')                        # correct for truncation of Hilbert space
    sys.print()                               # view parameters
    sys.plot('spectrum')                      # plot energy eigenvalues
    sys.save()                                # save data
    sys = System(1,1,40,2, indiv=True)        # generate a new system with custom parameters
    sys.select([[0],[0]])                     # select the ground state for each parity
    sys.plot('spins')                         # plot ⟨J_z⟩ for each individual spin
    sys = sys.load()                          # load the previous system
    uncertainty(sys.J_x, sys.states, sys=sys) # plot ΔJ_x """

""" Data descriptions

    Individual states
    -----------------
    Use the following as an example for the Dicke model, where field_dim=3, j=1/2, N=ω=ω0=1, and λ=0:
        |n,  m_J⟩     state array
        |0,  1/2⟩     [1. 0. 0. 0. 0. 0.]            
        |0, -1/2⟩     [0. 1. 0. 0. 0. 0.]            
        |1,  1/2⟩     [0. 0. 1. 0. 0. 0.]            
        |1, -1/2⟩     [0. 0. 0. 1. 0. 0.]            
        |2,  1/2⟩     [0. 0. 0. 0. 1. 0.]            
        |2, -1/2⟩     [0. 0. 0. 0. 0. 1.]            
    Similarly, for N=2:
        |0,    1⟩     [1. 0. 0. 0. 0. 0. 0. 0. 0.]   
        |0,    0⟩     [0. 1. 0. 0. 0. 0. 0. 0. 0.]   
        |0,   -1⟩     [0. 0. 1. 0. 0. 0. 0. 0. 0.]   
        |1,    1⟩     [0. 0. 0. 1. 0. 0. 0. 0. 0.]   
        |1,    0⟩     [0. 0. 0. 0. 1. 0. 0. 0. 0.]   
        |1,   -1⟩     [0. 0. 0. 0. 0. 1. 0. 0. 0.]   
        |2,    1⟩     [0. 0. 0. 0. 0. 0. 1. 0. 0.]   
        |2,    0⟩     [0. 0. 0. 0. 0. 0. 0. 1. 0.]   
        |2,   -1⟩     [0. 0. 0. 0. 0. 0. 0. 0. 1.]   
    Hence, the first N+1 entries correspond to the vacuum state for each possible m_J value.
    The second N+1 entries correspond to one excited state for each possible m_J value.
    The function sort_by_quantum_numbers() can be used to determine initial values for |n, m_J⟩.
    
    Sets of states
    --------------
    Plotting things as a function of λ requires sets of states. These currently have the following structure.
        states       : list(2D_eigenvalue_array, 2D_eigenvector_array)
                       each "row" in these arrays corresponds to a particular λ value
                       each column corresponds to an individual state, where there are field_dim*m_J_max states altogether
                       each entry in the "2D" eigenvector array corresponds to an eigenvector   
    For example, this is some states[0] with field_dim=2, j=1/2, and N=1 for two λ values.
        [[-5.0 -4.9  5.0  5.1]
         [-5.5 -5.6  5.7  5.6]]
    This is the corresponding states[1] for the same set of parameters.
        [[[ 0.   0.   1.   0. ]
          [ 1.   0.   0.   0. ]
          [ 0.   0.   0.   1. ]
          [ 0.   1.   0.   0. ]]
         [[ 0.2  0.   0.   0.9]
          [ 0.  -0.9  0.2  0. ]
          [ 0.   0.2  0.9  0. ]
          [-0.9  0.   0.   0.2]]]
    The 1D array states[0][0] shows the eigenvalues for the first λ.
    The 2D array states[1][0] shows the eigenvectors for the first λ.
    The eigenvalue entry at states[0][0][2] is 5.0 and corresponds to the eigenvector column at states[1][0][:,2].
    The eigenvalue entry at states[0][1][2] is 5.7 and corresponds to the eigenvector column at states[1][1][:,2]. """

########################################################################################################################################################
# Imports
## Utility
import matplotlib.pyplot as plt               # plotting
from matplotlib.gridspec import GridSpec      # plotting
from mpl_toolkits.mplot3d import Axes3D       # plotting
from tqdm import tqdm                         # progress bars
import pickle                                 # file saving

## Computation
import numpy as np                            # tensor algebra
from scipy.stats import norm                  # vector normalization
from scipy.linalg import expm                 # unitary transformations
from scipy.special import jv                  # Chebyshev algorithm; Bessel function of the first kind
from scipy.sparse import identity, csr_matrix # Chebyshev algorithm

########################################################################################################################################################
# Operators
def _create_J_operators(sys, j_set=None):
    """ Generates and returns angular momentum matrices in the spin space.
        
        Parameters
        ----------
        j_set : int or half-int; manually sets collective spin """
    
    # Set dimensions manually or import from initialized system
    if j_set:
        dim = int(round(2 * j_set + 1))
        j   = sys.N * j_set
    elif sys.indiv:
        dim = 2**sys.N
    else:
        dim = sys.spin_dim
        j   = sys.J
    
    # Construct and return operators in the collective space (2J+1)
    if not sys.indiv:
    
        # Construct ladder operators
        J_p    = np.zeros((dim, dim))
        J_m    = np.zeros((dim, dim))
        m_vals = np.arange(j, -(j+1), -1)
        for i in tqdm(range(dim - 1), desc=f"{'creating J operators':<35}"):
            m           = m_vals[i]
            val         = sys.ℏ * np.sqrt(j*(j+1)-m*(m-1))
            J_p[i, i+1] = val
            J_m[i+1, i] = val

        # Construct component operators
        J_x = (1/2) *(J_p + J_m)
        J_y = (1/2j)*(J_p - J_m)
        J_z = sys.ℏ * np.diag([j-m for m in range(dim)])
        
        return {'J_x': J_x, 'J_y': J_y, 'J_z': J_z, 'J_m': J_m, 'J_p': J_p}
    
    # Construct operators in full spin space (2^N)
    else:
    
        # Initialize empty matrices
        J_x = np.zeros((dim, dim), dtype=complex)
        J_y = np.zeros((dim, dim), dtype=complex)
        J_z = np.zeros((dim, dim), dtype=complex)
        J_x_list, J_y_list, J_z_list = [], [], []
        J_p_list, J_m_list = [], []
        
        # Sort through each atom in the system
        for i in range(sys.N):
            J_x_cache = 1
            J_y_cache = 1
            J_z_cache = 1
            
            # Sort through all the other spins
            for k in range(sys.N):
                if k == i:
                    J_x_cache = np.kron(J_x_cache, (sys.ℏ/2) * np.array([[0, 1],   [1, 0]]))
                    J_y_cache = np.kron(J_y_cache, (sys.ℏ/2) * np.array([[0, -1j], [1j, 0]]))
                    J_z_cache = np.kron(J_z_cache, (sys.ℏ/2) * np.array([[1, 0],   [0, -1]]))
                else:
                    J_x_cache = np.kron(J_x_cache, np.eye(2))
                    J_y_cache = np.kron(J_y_cache, np.eye(2))
                    J_z_cache = np.kron(J_z_cache, np.eye(2))
            J_x_list.append(np.kron(np.eye(sys.field_dim), J_x_cache))
            J_y_list.append(np.kron(np.eye(sys.field_dim), J_y_cache))
            J_z_list.append(np.kron(np.eye(sys.field_dim), J_z_cache))
            
            J_x += J_x_cache
            J_y += J_y_cache
            J_z += J_z_cache
        
        J_m = (J_x - 1j * J_y) / 2
        J_p = (J_x + 1j * J_y) / 2
        sys.J_x_list, sys.J_y_list, sys.J_z_list = J_x_list, J_y_list, J_z_list
        sys.J_p_list, sys.J_m_list = J_p_list, J_m_list
        
        return {'J_x': J_x, 'J_y': J_y, 'J_z': J_z, 'J_m': J_m, 'J_p': J_p}
    
    # Return identity if needed
    if sys.N == 0:
        return {'J_x': np.eye(dim), 'J_y': np.eye(dim), 'J_z': np.eye(dim), 'J_m': np.eye(dim), 'J_p': np.eye(dim)}

def _create_a_operators(sys):
    """ Generates creation and annihilation matrices in the Fock basis.
        
        Returns
        -------
        a     : matrix; creation operator for photon field
        a_dag : matrix; annihilation operator for photon field """
    
    a = np.zeros((sys.field_dim, sys.field_dim))
    
    for i in tqdm(range(1, sys.field_dim), desc=f"{'creating a operators':<35}"):
        a[i-1, i] = np.sqrt(i)
    
    a_dag = a.conj().T
    return {'a': a, 'a_dag': a_dag}

def _create_excitation_operators(sys):
    """ Generate total excitation and parity operators. """
    
    # Total excitation number
    N = sys.N/2 * np.eye(sys.J_z.shape[0])
    η = tolerance(sys.a_dag@sys.a + sys.J_z + N)
    
    # Parity
    field_parity = expm(1j * np.pi * sys.a_dag_field @ sys.a_field)
    if sys.mod != 'SOC': spin_parity  = expm(1j * np.pi * sys.J_z_spin)
    else:                spin_parity  = expm(1j * np.pi * sys.J_x_spin)
    P = tolerance(np.kron(field_parity, spin_parity))
    
    return {'η': η, 'P': P}

def _compute_tensor_products(sys):
    """ Takes the tensor product of the field and atom operators and yields the full Hamiltonian. """
    
    a     = np.kron(sys.a_field,     np.eye(sys.J_z_spin.shape[0]))
    a_dag = np.kron(sys.a_dag_field, np.eye(sys.J_z_spin.shape[0]))
    
    J_x   = np.kron(np.eye(sys.a_field.shape[0]), sys.J_x_spin) 
    J_y   = np.kron(np.eye(sys.a_field.shape[0]), sys.J_y_spin) 
    J_z   = np.kron(np.eye(sys.a_field.shape[0]), sys.J_z_spin)
    J_p   = np.kron(np.eye(sys.a_field.shape[0]), sys.J_p_spin)
    J_m   = np.kron(np.eye(sys.a_field.shape[0]), sys.J_m_spin)
    
    return {'J_x': J_x, 'J_y': J_y, 'J_z': J_z, 'J_m': J_m, 'J_p': J_p, 'a': a, 'a_dag': a_dag}

########################################################################################################################################################
# Operations
def expectation(operator, states, sys=None, rounded=False, title=None):
    """ Calculates and returns (or plots) expectation values for one or more states. 
        
        Parameters
        ----------
        operator          : 2D array
        states            : standard representation or column vector
        sys               : commandline use; plots directly 
        
        Returns
        -------
        expectation_value : float
        expectation_array : 2D array; one row per λ

        Examples
        --------
        expectation(sys.J_z, sys.states)
        expectation(sys.a, rtc(sys.states[1][0][:,0])) """
    
    # Process states matrix
    if type(states) == list:
        expectation_array = []
        
        # Sort through trials
        for i in tqdm(range(states[1].shape[0]), desc=f"{'calculating expectation values':<35}"):
            temp_list_1 = []

            # Sort through states
            for j in range(states[1][i].shape[1]):
                temp_list_2 = expectation(operator, rtc(states[1][i][:,j]))
                temp_list_1.append(temp_list_2)
            expectation_array.append(np.array(temp_list_1).T)
        
        # Convert to array and round to nearest integer if desired
        if rounded: expectation_array = discretize(np.array(expectation_array))
        else:       expectation_array = np.array(expectation_array)
        
        # Return results or plot with system variable
        if not sys:
            return np.array(expectation_array)
        else:
            if not title: title = 'f(λ)'
            plot_list = [[(f"$λ$", f"⟨${title}$⟩"), (sys.var, np.array(expectation_array)), (0, 0), ('plot')]]
            plot_handling(plot_list)
    
    # Process column vector
    else:
        expectation_value = np.conj(states).T @ operator @ states
        return np.real(expectation_value.item())

def uncertainty(operator, states, sys=None):
    """ Calculates and returns (or plots) expectation values for one or more states. 
        
        Parameters
        ----------
        operator     : 2D array
        states       : states matrix or column vector
        sys          : commandline use; plots directly 
        
        Returns
        -------
        output       : 2D array; one row per λ 
        
        Example
        -------
        uncertainty(sys.J_z, sys.states, sys=sys) """

    # Initialize data containers
    expectations, output = [[], []], []

    # Calculate expectation values
    expectations[0] = expectation(operator,            states)
    expectations[1] = expectation(operator @ operator, states)

    # Use expectation values to calculate uncertainty
    for i in range(states[1].shape[0]):
        cache = []
        for j in range(states[1].shape[2]):
            cache.append(np.sqrt(abs(expectations[1][i][j]-expectations[0][i][j]**2)))
        output.append(cache)
    
    # Return results or plot with system variable
    if not sys:
        return np.array(output)
    else:
        plot_list = [[(f"$λ$", f"$f(λ)$"), (sys.var, np.array(output)), (0, 0), ('plot')]]
        plot_handling(plot_list)

def partial_trace(ρ, dim_A, dim_B, subsystem):
    """ Computes the partial trace of a matrix.

        Parameters
        ----------
        ρ         : 2D array; density matrix 
        dim_A     : integer; dimension of subsystem A
        dim_B     : integer; dimension of subsystem B
        subsystem : string in {'A', 'B'}; subsystem to be traced out

        Returns
        -------
        ρ_reduced : reduced matrix after the partial trace

        Example
        -------
        state = rtc(sys.states[1][i][:,j])
        ρ     = np.outer(state, state.conj())
        partial_trace(ρ, sys.field_dim, sys.spin_dim, 'A') """
    
    ρ = ρ.reshape((dim_A, dim_B, dim_A, dim_B))
    if subsystem == 'B':
        ρ_reduced = np.trace(ρ, axis1=1, axis2=3)
    elif subsystem == 'A':
        ρ_reduced = np.trace(ρ, axis1=0, axis2=2)
    return ρ_reduced

def partial_transpose(ρ, dim_A, dim_B, subsystem):
    """ Perform the partial transposition of a bipartite density matrix.
        
        Parameters
        ----------
        ρ         : 2D array; density matrix 
        dim_A     : integer; dimension of subsystem A
        dim_B     : integer; dimension of subsystem B
        subsystem : string in {'A', 'B'}; subsystem to be traced out
        
        Returns
        -------
        eigensum  : float; sum of the absolute value of each eigenvalue
        
        Example
        -------
        state = rtc(sys.states[1][i][:,j])
        ρ     = np.outer(state, state.conj())
        partial_transpose(ρ, sys.field_dim, sys.spin_dim, 'A') """
    
    # Reshape ρ to handle subsystems separately
    ρ = ρ.reshape((dim_A, dim_B, dim_A, dim_B))
    
    # Perform partial transposition
    if subsystem == 'A':
        ρ = np.transpose(ρ, (2, 1, 0, 3))
    elif subsystem == 'B':
        ρ = np.transpose(ρ, (0, 3, 2, 1))
    
    # Return to original shape
    ρ = ρ.reshape((dim_A * dim_B, dim_A * dim_B))
    
    # Check for entanglement by computing the eigenvalues
    eigenvalues = np.linalg.eigvals(ρ)
    eigensum    = 0
    for i in range(len(eigenvalues)):
        eigensum += abs(eigenvalues[i])
    
    return [ρ, eigensum]

def mean_field(sys, state):
    
    # Data processing
    def rk4_step(f, y, t, dt):
        """Performs a single step of the RK4 method for complex-valued ODEs."""
        k1 = f(y, t)
        k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(y + dt * k3, t + dt)
        return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def solve_rk4(f, y0, t_span, dt):
        """Solves a system of ODEs using the RK4 method for complex variables."""
        t_values = np.arange(t_span[0], t_span[1] + dt, dt)
        y_values = np.zeros((len(t_values), len(y0)), dtype=complex)  # Ensure complex dtype
        
        y = np.array(y0, dtype=complex)  # Ensure initial conditions are complex
        
        for i, t in enumerate(t_values):
            y_values[i] = y
            y = rk4_step(f, y, t, dt)
        
        return t_values, y_values

    def safe_complex_multiply(a, b, max_magnitude=1e10):
        """
        Safely multiply complex numbers with magnitude limiting
        """
        result = a * b
        
        # Limit magnitude to prevent overflow
        mag = np.abs(result)
        if mag > max_magnitude:
            result *= max_magnitude / mag
        
        # Replace NaN with zero
        if np.isnan(result):
            return 0j
        
        return result

    # Set coupled equations
    def derivatives(y, t):
        """Defines the system of 8 coupled differential equations with safety checks."""
        dydt = np.zeros(4, dtype=complex)
        
        dydt[0] = -1j * (sys.ω * y[0] + (sys.var[0] / np.sqrt(sys.N)) * (np.conj(y[2]) + y[2]))             # a
        dydt[1] = -1j * (sys.var[0] / np.sqrt(sys.N)) * (np.conj(y[0]) + y[0]) * (np.conj(y[2]) - y[2])     # Z
        dydt[2] = -1j * (sys.ω0 * y[2] - 2 * (sys.var[0] / np.sqrt(sys.N)) * (np.conj(y[0]) + y[0]) * y[1]) # M
        dydt[3] = -1j * (sys.var[0] / np.sqrt(sys.N)) * (np.conj(y[2]) + y[2]) * (np.conj(y[0]) - y[0])     # a†a
        
        return dydt

    # Plot results
    def plot_complex_variables(t_values, y_values, variable_labels):
        """
        Create three subplots: real parts, magnitudes, and phases of complex variables
        
        Parameters:
        - t_values: Time array
        - y_values: Complex variable values
        - variable_labels: Labels for each variable
        """
        
        # Colors for distinguishing variables    
        colors = [
            'darkred',
            'orange',
            
            'darkblue',
            'green',
            
            'black',
            'darkgray']
        
        linestyles = [
            'dashed',
            'dashed',
            
            'solid',
            'solid',
            
            'dotted',
            'dotted']
        
        # Create a figure with three rows of subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 7))
        
        # Labels
        parameters = f"($ω, ω_0, λ, N$) = {sys.ω, sys.ω0, sys.var[0], sys.N}"
        initial_conditions = f"($a_0, Z_0, M_0, n_0$) = {np.round(y0[0], 1), np.round(y0[1], 1), np.round(y0[2], 1), np.round(y0[3], 1)}"
        plt.suptitle(f"{parameters + '      ' + initial_conditions}")
        
        ax1.set_title(r'Spin Polarization $⟨J_z⟩$')
        ax1.set_ylabel(r'arb. units')
        
        ax2.set_title('Destruction Operators (Magnitude)')
        ax2.set_ylabel(r'arb. units')
        
        ax3.set_title('Destruction Operators (Phase)')
        ax3.set_ylabel(r'$2πn$')
        
        ax4.set_title('Occupation Number $⟨a^†a⟩$')
        ax4.set_ylabel('$n$')
        ax4.set_xlabel('time (arb. units)')
        
        ## Plot observables
        ax1.plot(t_values, y_values[:, 1].real, label=f'{variable_labels[1]}', color=colors[1 % len(colors)], linestyle=linestyles[1 % len(colors)])
        
        ## Plot non-Hermitian magnitudes
        ax2.plot(t_values, np.abs(y_values[:, 0]), label=variable_labels[0], color=colors[0 % len(colors)], linestyle=linestyles[0 % len(colors)])
        ax2.plot(t_values, np.abs(y_values[:, 2]), label=variable_labels[2], color=colors[2 % len(colors)], linestyle=linestyles[2 % len(colors)])
        
        ## Plot non-Hermitian phases
        phases_0 = np.unwrap(np.angle(y_values[:, 0]))
        phases_1 = np.unwrap(np.angle(y_values[:, 2]))

        ### Get min and max phase values to determine range for grid lines
        phase_min = min(phases_0.min(), phases_1.min())
        phase_max = max(phases_0.max(), phases_1.max())

        ### Generate integer multiples of 2π within the range
        grid_lines = np.arange(
            np.floor(phase_min / (2*np.pi)) * 2*np.pi, 
            np.ceil(phase_max / (2*np.pi)) * 2*np.pi + 1, 2*np.pi)

        ### Plot phase data
        ax3.plot(
            t_values, phases_0, label=variable_labels[0], 
            color=colors[0 % len(colors)], linestyle=linestyles[0 % len(colors)])
        ax3.plot(
            t_values, phases_1, label=variable_labels[2], 
            color=colors[2 % len(colors)], linestyle=linestyles[2 % len(colors)])

        ### Add horizontal lines at each integer multiple of 2π
        for line in grid_lines:
            ax3.axhline(line, color='whitesmoke', linewidth=0.7)

        ### Set tick locations and labels to multiples of 2π
        ax3.set_yticks(grid_lines)
        ax3.set_yticklabels([rf"${int(k // (2*np.pi))}$" for k in grid_lines])  # LaTeX-style labels

        ## Plot occupation number
        ax4.plot(t_values, y_values[:, 3].real, label=r'$a^†a$', color='purple')
        
        # Add legends
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        return fig

    # Find initial conditions
    y0    = np.zeros(4, dtype=complex)
    y0[0] = expectation(sys.a, state)           # a
    y0[1] = expectation(sys.J_z, state)         # J_z
    y0[2] = expectation(sys.J_m, state)         # J_-
    y0[3] = expectation(sys.a_dag@sys.a, state) # a†a
    
    ## Parameters (time)
    t_span = (0, 10)   # Time range
    dt     = 0.01      # Time step

    # Solve the system
    t_values, y_values = solve_rk4(derivatives, y0, t_span, dt)

    # Labels for the variables
    variable_labels = [r'$a$', r'$Z$', r'$M$']

    # Create and show the real parts, magnitude, and phase plots
    plot_complex_variables(t_values, y_values, variable_labels)
    plt.show()

########################################################################################################################################################
# Quick plots
def _occupation(sys):
    """ Prepare ⟨n⟩ and ⟨J_z⟩ for plotting.

        Returns
        -------
        plot_list : list of tuples; holds data to be used with plot_handling """
    
    # Calculate expectation values
    n_expectations   = expectation(sys.a_dag @ sys.a, sys.states)
    J_z_expectations = expectation(sys.J_z,           sys.states)
    
    # Construct and return plot list
    plot_list = [[(f"$λ$", f"$⟨n⟩$"),   (sys.var, n_expectations),   (0, 0), ('plot')],
                 [(f"$λ$", f"$⟨J_z⟩$"), (sys.var, J_z_expectations), (0, 1), ('plot')]]
    return plot_list

def _spectrum(sys):
    """ Prepare energy eigenvalues for plotting.

        Returns
        -------
        plot_list : list of tuples; holds data to be used with plot_handling 
    
        Optional
        --------
        Subtract the ground state energy from all eigenvalues
            for i in range(len(states[0])):
                for j in range(len(states[0][0])): 
                    val = states[0][i].copy()[0]
                    states[0][i][-(j+1)] -= val """
    
    plot_list = [[(f"$λ$", f"$E$"), (sys.var, sys.states[0]), (0, 0), ('plot')]]
    return plot_list

def _entropy(sys):
    """ Computes von Neumann entropy and partial transposition for plotting.

        Returns
        -------
        plot_list : list of tuples; holds data to be used with plot_handling """

    def _von_Neumann(ρ, base=2):
        """ Calculate the von Neumann entropy via S(ρ) = -Tr(ρ log ρ).
        
            Parameters
            ----------
            ρ       : 2D array; density matrix
            base    : float; base of the logarithm
            
            Returns
            -------
            entropy : float; von Neumann entropy """
        
        eigvals = np.linalg.eigvalsh(ρ)
        eigvals = eigvals[eigvals > 0] # avoids issues with log(0)
        
        if base == 2:
            log_eigvals = np.log2(eigvals)
        elif base == np.e:
            log_eigvals = np.log(eigvals)
        else:
            log_eigvals = np.log(eigvals) / np.log(base)
        
        return -np.sum(eigvals * log_eigvals)

    # Prepare entropy arrays
    entropy_tot   = np.zeros_like(sys.states[0])
    entropy_field = np.zeros_like(sys.states[0])
    entropy_spin  = np.zeros_like(sys.states[0])
    eigensum_spin = np.zeros_like(sys.states[0])
    
    # Sort through trials
    for i in tqdm(range(sys.states[1].shape[0]), desc=f"{'calculating entropy':<35}"):
    
        # Sort through states:
        for j in range(sys.states[1][i].shape[1]):
            
            # Extract state and compute density matrix
            state = rtc(sys.states[1][i][:,j])
            ρ     = np.outer(state, state.conj())
            
            # Compute reduced density matrices
            ρ_spin  = partial_trace(ρ, sys.field_dim, sys.spin_dim, subsystem='A')
            
            # Compute bipartite negativity
            eigensum = partial_transpose(ρ, sys.field_dim, sys.spin_dim, subsystem='A')[1]
            
            # Calculate the von Neumann entropy for the total system and subsystems
            entropy_field[i][j] = _von_Neumann(ρ_spin,  base=2)
            eigensum_spin[i][j] = (eigensum - 1)/2
    
    plot_list = [[(f"$λ$", f"von Neumann entropy"), (sys.var, entropy_field), (0, 0), ('plot')],
                 [(f"$λ$", f"partial transpose"),   (sys.var, eigensum_spin), (0, 1), ('plot')]]
    return plot_list

def _squeezing(sys):
    """ Calculates the standard deviation of a given operator and a set of states.

        Returns
        -------
        plot_list : list of tuples; holds data to be used with plot_handling """

    # Initialize data containers
    expectations, output = [[], [], []], []
    
    # Calculate expectation values
    expectations[0] = expectation(sys.a_dag @ sys.a, sys.states)
    expectations[1] = expectation(sys.a,             sys.states)
    expectations[2] = expectation(sys.a @ sys.a,     sys.states)
    
    # Use expectation values to calculate uncertainty
    for i in range(len(expectations[0])):
        cache = []
        for j in range(len(list(expectations[0][i]))):
            factor = 1 + 2*(expectations[0][i][j] - abs(expectations[1][i][j])**2 - abs(expectations[2][i][j] - expectations[1][i][j]**2))
            cache.append(factor)
        output.append(cache)
    return np.array(output)

def _Lindbladian(sys, custom=False):
    """ Evolves and plots the current states over time with a jump operator. """
    
    # Check for a reasonable runtime
    if sys.states[1].shape[2] >= 5:
        confirm = print(f"{'Warning! Many states detected. Process anyways? [y/n]':<35}: ")
        if not confirm: return
    
    # Define the algorithm
    def _euler_time_evolution(ρ, jump_operators, index):
        
        # Prepare data
        H = sys.H_list[index]
        
        # Construct the Lindbladian and evolve the density matrix
        dρ = -1j * (H @ ρ - ρ @ H)                                       # [H, ρ]
        for L in jump_operators:
            anticommutator = (L.conj().T @ L) @ ρ + ρ @ (L.conj().T @ L) # {L†L, ρ}
            dρ += L @ (ρ @ L.conj().T) - (1/2) * anticommutator          # LρL†-{L†L, ρ}/2
        
        return ρ + dt * dρ
    
    # Initialize parameters and data container
    if custom:
        t_max      = float(input(f"{'total time (ex. 5)':<35}: "))
        t_shift    = float(input(f"{'initial time (ex. 0)':<35}: "))
        jump       = input(f"{'jump operators (ex. a, J_z)':<35}: ").split(',')
        observable = input(f"{'observable (ex. J_z)':<35}: ")
    
    else:
        t_max      = 5
        t_shift    = 0
        jump       = ['a']
        observable = 'n'
    
    dt = (t_max-t_shift)/1000
    times = np.linspace(t_shift, t_max+t_shift, int((t_max-t_shift)/dt))
    operator_dict = {
        'E':     None,
        'a':     sys.a,
        'a_dag': sys.a_dag,
        'n':     sys.a_dag@sys.a,
        'J_z':   sys.J_z,
        'I':     np.eye(sys.a.shape[0])}
    jump_operators = [operator_dict[key] for key in jump]
    observable_name = observable
    observable = operator_dict[observable]
    plot_list = []
    
    # Sort through each trial
    for i in range(len(sys.var)):

        # Initialize data container for plotting
        expectation_values = []

        # Generate density matrix for each trial
        ρ_array = []
        for j in range(sys.states[1][i].shape[1]):
            ρ_array.append(np.outer(sys.states[1][i][:,j], sys.states[1][i][:,j].conj()))
        ρ_array = np.array(ρ_array)

        # Sort through each state
        for j in range(len(ρ_array)):
            ρ = ρ_array[j]
            expectation_values.append([])

            # Sort through each time step
            for _ in tqdm(times, desc=f"{'calculating Lindbladian':<35}"):
                if observable_name == 'E': observable = sys.H_list[i]
                expectation_values[j].append(np.real(np.trace(observable @ ρ)))
                ρ = _euler_time_evolution(ρ, jump_operators, i)
        
        plot_list.append([
            (f"$t, λ={round(sys.var[i],2)}$", f"$⟨{observable_name}⟩$"),
            (times, np.array(expectation_values).T),
            (0, i),
            ('plot')])
    
    return plot_list

def _Chebyshev(sys, custom=False):
    """ Estimates time evolution by utilizing Chebyshev recursion relations. """
    
    # Check for a reasonable runtime
    if sys.states[1].shape[2] >= 5:
        confirm = print(f"{'Warning! Many states detected. Process anyways? [y/n]':<35}: ")
        if not confirm: return
    
    # Define the algorithm
    def _chebyshev_time_evolution(input_state, time, num_terms, index):
        
        # Bug fix to handle complex numbers
        psi_cache = np.zeros_like(input_state, dtype=np.complex128)
        input_state = psi_cache + input_state
        
        # Prepare data
        H     = sys.H_list[index]
        E_min = sys.states_backup[0][index].max().real
        E_max = sys.states_backup[0][index].min().real

        # Scale the Hamiltonian by minimum and maximum energy eigenvalues
        H_scaled = (2 * H - (E_max + E_min) * csr_matrix(identity(H.shape[0]))) / (E_max - E_min)
        
        # Calculate the zeroth-order term
        T_0 = input_state
        T_1 = H_scaled @ input_state
        order, argument = 0, (E_max - E_min) * time / 2
        output_state = jv(order, argument) * T_0
        
        # Iteratively compute higher-order terms
        for i in range(1, num_terms):
            
            # Add the contribution of the nth term
            Tn = np.zeros(T_0.shape[0], dtype=np.complex128).reshape((input_state.shape[0], 1))
            Tn += 2 * (H_scaled @ T_1) - T_0
            order, argument = i, (E_max - E_min) * time / 2
            output_state += 2*(-1j)**i * jv(order, argument) * Tn
            
            # Prepare for the next iteration
            T_0, T_1 = T_1, Tn
        
        return output_state
    
    # Initialize parameters and data container
    if custom:
        num_terms  = int(input(f"{'iteration depth (ex. 7)':<35}: "))
        t_max      = int(input(f"{'total time (ex. 5)':<35}: "))
        t_shift    = int(input(f"{'initial time (ex. 0)':<35}: "))
        observable = input(f"{'observable (ex. J_z)':<35}: ")

    else:
        num_terms  = 7
        t_max      = 5
        t_shift    = 0
        observable = 'n'
    
    dt = (t_max-t_shift)/1000
    times = list(np.linspace(t_shift, t_max+t_shift, int((t_max-t_shift) / dt)))
    operator_dict = {
        'E':     None,
        'a':     sys.a,
        'a_dag': sys.a_dag,
        'n':     sys.a_dag@sys.a,
        'J_z':   sys.J_z,
        'I':     np.eye(sys.a.shape[0])}
    observable_name = observable
    observable = operator_dict[observable]
    plot_list = []
    
    # Cycle through trials
    for i in range(sys.states[1].shape[0]):
        expectation_values = []
        
        # Cycle through states
        for j in range(sys.states[1][i].shape[1]):
            expectation_values.append([])
            state = rtc(sys.states[1][i][:,j])
            
            # Run the algorithm
            for time in tqdm(times, desc=f"{'calculating evolution':<35}"):
            
                # Compute the time-evolved state
                state_evolved = _chebyshev_time_evolution(state, time, num_terms, i)
                
                # Calculate a property of the evolved state (e.g., probability |state_evolved|^2)
                if observable_name == 'E': observable = sys.H_list[i]
                measure = expectation(observable, state_evolved)
                
                # Store the total measure at this time step (or any other property of interest)
                expectation_values[j].append(measure.real)
        
        # Prepare data for plotting
        expectation_values = np.array(expectation_values).T
        plot_list.append([(f"$t,\tλ={round(sys.var[i],2)}$", f"$⟨{observable_name}⟩$"), 
                          (times, expectation_values), 
                          (0, i), 
                          ('plot')])
    
    return plot_list

def _E_spacing(sys):
    """ Computes and plots level spacing statistics for a 2D NumPy array of sorted energy eigenvalues,
        overlaying Poisson and Wigner-Dyson distributions. """
    
    A, B = sys.states[0].shape  # A: number of trials, B: number of eigenvalues per trial
    
    # Initialize plot
    fig, axes = plt.subplots(3, 1, figsize=(6, 3), sharex=True)
    if A == 1: axes = [axes] # ensure axes is iterable even if A=1
    
    # Initialize distributions for comparison to data
    s_vals       = np.linspace(0, 3, 100)
    Poisson      = np.exp(-s_vals)
    Wigner_Dyson = (np.pi/2) * s_vals * np.exp(-(np.pi/4) * s_vals**2)
    
    # Process first, middle, and last coupling strength
    for i in [0, A//4, A//2]:
        if i == 0:      j = 0
        elif i == A//4: j = 1
        else:           j = 2
        
        # Compute and normalize level spacings for the current trial
        spacings = np.diff(sys.states[0][i])
        mean_spacing = np.mean(spacings)
        normalized_spacings = spacings / mean_spacing

        # Construct histogram of level spacings
        axes[j].hist(
            x       = normalized_spacings,
            bins    = 100,
            density = True,
            alpha   = 0.6,
            color   = 'b',
            label   = f"λ={round(sys.var[i])}")

        # Overlay theoretical distributions
        axes[j].plot(s_vals, Poisson,      'r--')
        axes[j].plot(s_vals, Wigner_Dyson, 'g-')
        axes[j].axis(xmin=0, xmax=4, ymin=0, ymax=1)
        axes[j].legend()
    
    axes[-1].set_xlabel("Normalized Level Spacing")
    plt.suptitle("Level Spacing Statistics")
    plt.show()

########################################################################################################################################################
# Utilities
def details():
    """ System
        ------
        System._Hamiltonian : Hamiltonian creator        : typically called elsewhere
        System.sort         : state sorter by eigenvalue : sometimes useful
        System.select       : state selector             : very useful
        System.plot         : some common plots          : very useful
        System.print        : system details             : sometimes useful

        Other
        -----
        examples          : use-case illustrator         : sometimes useful
        expectation       : data calculator and plotter  : very useful
        uncertainty       : data calculator and plotter  : sometimes useful
        partial_trace     : density matrix processor     : typically called elsewhere
        partial_transpose : density matrix processor     : typically called elsewhere
        plot_handling     : GridSpec processor           : typically called elsewhere
        rtc               : state processor              : very useful
        tolerance         : tensor processor             : sometimes useful
        discretize        : tensor processor             : sometimes useful """
    pass

def examples(index):
    """ Run a preset example.
        
        Options
        -------
        1 : parameters, variables, and spectrum
        2 : state selection and time evolution
        3 : squeezing and uncertainty
        4 : Dicke order parameter
        5 : Tavis-Cummings ground state """
    
    # Tutorial 1: Setting parameters
    if index == 1:
        
        # Description
        print("\nTutorial: System creation and quick plotting.\n")
    
        # Initialize the parameters
        ω  = 1  # set field frequency
        ω0 = 1  # set spin frequency
        n  = 40 # set field truncation
        N  = 4  # set number of atoms
        
        # Initialize the variable
        initial = 0   # set scaling of initial coupling strength
        final   = 2   # set scaling of final coupling strength
        trials  = 101 # set number of datapoints
        var_set = [initial, final, trials] # create variable
        
        # Create and solve the Hamiltonian
        sys = System(ω, ω0, n, N, var_set=var_set)
        
        # Make a calculation
        sys.plot('spectrum')
    
    # Tutorial 2: Selecting states
    elif index == 2:
        
        # Description
        print("\nTutorial 2: State selection and time evolution.\n")
    
        # Initialize system
        sys = System(1, 1, 10, 4, var_set=[0.1, 4, 21])

        # Select a superposition of the lowest eigenstate of each parity
        sys.select([[0], [0]])
        sys.select('super')

        # Make a calculation
        sys.plot('Lindblad')

    # Example 1: Squeezing and uncertainty
    elif index == 3:
        
        # Description
        print("\nExample 1: Squeezing and uncertainty.\n")
        
        # Initialize system
        sys = System(0.1,10,40,4, var_set='standard')

        # Select specific eigenstates
        sys.select([0, 1, 2])
        
        # Make a calculation
        sys.plot('squeezing')
    
    # Example 2: Dicke order parameter
    elif index == 4:
        
        # Description
        print("\nExample 2: Bifurcation of the order parameter.\n")
        
        # Initialize system
        sys = System(1,1,40,4, var_set=[0.1, 10, 101], mod='pert')
        sys.sort('E')
        sys.select([0, 1])
        
        # Calculate and plot
        expectation(sys.a, sys.states, sys=sys, title='a')
    
    # Example 3: Tavis-Cummings ground state 
    elif index == 5:
        
        # Description
        print("\nExample 3: Tavis-Cummings ground state.\n")
    
        # Initialize system
        sys = System(1,1,10,2, var_set=[0.1,10,101], mod='TC', indiv=True)
        sys.sort('η', 'E')
        sys.select([0,1,4,8,12,16,20])
        
        # Calculate and plot
        η_vals = expectation(sys.η, sys.states, rounded=True)
        E_plot = [(f"$λ$", f"$⟨E⟩$"), (sys.var, sys.states[0]), (0, 0), ('plot')]
        η_plot = [(f"$λ$", f"$⟨η⟩$"), (sys.var, η_vals),        (0, 1), ('plot')]
        plot_handling([E_plot, η_plot], sys.numbers)
    
    else: print('There are no examples with this value.')
    return sys

def plot_handling(results, numbers=None, plot_mode='2D', no_show=False, sys=None):
    """ Initializes matplotlib and generates plots from input data.
        
        Parameters
        ----------
        results   : list of tuples; see Details and Example
        numbers   : 3D array; corresponding quantum numbers for coloring plots
        plot_mode : string in {'2D', '3D'}; self-explanatory
        no_show   : bool; omits plt.show() if True
        sys       : optional inclusion of System object; only used for 3D plotting
      
        Details
        -------
        titles    : (row_title, column_title)
        values    : (x_values,  y_values)
        indices   : (row_index, column_index
        style     : (plot_type)
        
        Example
        -------
        n_vals = expectation(sys.a_dag @ sys.a, sys.states)
        J_vals = expectation(sys.J_z,           sys.states)
        E_plot = [(f"$λ$", f"$⟨E⟩$"),   (sys.var, sys.states[0]), (0, 1), ('plot')]
        n_plot = [(f"$λ$", f"$⟨n⟩$"),   (sys.var, n_vals),        (1, 0), ('plot')]
        J_plot = [(f"$λ$", f"$⟨J_z⟩$"), (sys.var, J_vals),        (1, 2), ('plot')]
        plot_handling([E_plot, n_plot, J_plot], sys.numbers) """
    
    # 2D plotting
    if plot_mode == '2D':
    
        # Initialize grid size based on 2D plot indices
        try:
            max_row = max(item[2][0] for item in results) + 1
            max_col = max(item[2][1] for item in results) + 1
        except:
            max_row = 1
            max_col = 1
        
        # Initialize figure with GridSpec
        fig = plt.figure(figsize=(3*max_col, 3*max_row))
        gs  = GridSpec(max_row, max_col, figure=fig)
        
        # Define color map
        color_map = {1: 'Reds_r', -1: 'Blues_r'}
        def get_colormap_color(cmap_name, column, num_columns):
            cmap = plt.get_cmap(cmap_name)
            normalized_value = column / num_columns
            capped_value = min(normalized_value, 0.9)
            return cmap(capped_value)  # Returns RGBA color

        # Loop through results for individual 2D plots
        for labels, values, indices, style in results:

            ax = fig.add_subplot(gs[indices[0], indices[1]])
            ax.set_xlabel(labels[0], fontsize=16)
            ax.set_ylabel(labels[1], fontsize=16)
            ax.ticklabel_format(useOffset=False)

            if numbers is not None:
                for i in range(len(values[1][0])):
                    color_category = color_map.get(int(numbers[0][i][0]))
                    set_color = get_colormap_color(color_category, i+1, len(values[1][0]))

                    if style == 'plot':
                        ax.plot(values[0], values[1][:, i], color=set_color)
                    elif style == 'contour':
                        ax.contourf(values[0], values[1][:, i], values[2], 100)
            else:
                if style == 'plot':
                    ax.plot(values[0], values[1])
                elif style == 'contour':
                    ax.contourf(values[0], values[1], values[2], 100)

        plt.tight_layout()
        if not no_show: plt.show()
    
    # 3D plotting
    elif plot_mode == '3D':
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, (labels, values, indices, style) in enumerate(results):
            x = values[0]
            y = np.full_like(values[1], sys.var[i])
            z = values[1]

            for trial in range(z.shape[1]):
                ax.plot(x, y[:, trial], z[:, trial])

        ax.set_xlabel(f"$t$", fontsize=12)
        ax.set_ylabel(f"$λ$", fontsize=12)
        ax.set_zlabel(labels[1], fontsize=12)
        if not no_show: plt.show()

def rtc(array):
    """ Shorthand to convert a row vector to a column vector (or vice-versa). 
    
        Example
        -------
        column_vector = rtc(sys.states[1][0][:,0]) """
    
    if len(array.shape) == 1:   vector = array.reshape(array.shape[0], 1)
    elif len(array.shape) == 2: vector = array.reshape(1, array.shape[0])
    return vector

def tolerance(array, zero_tolerance=1e-20):
    """ Sends each value in an array to zero if it is less than a set tolerance. """

    # Test real and imaginary parts
    real_part = np.where(np.abs(array.real) <= zero_tolerance, 0, array.real)
    imag_part = np.where(np.abs(array.imag) <= zero_tolerance, 0, array.imag)

    # Convert to a real-valued array if the imaginary part is null
    if np.all(imag_part == 0): return real_part
    else:                      return real_part + 1j * imag_part

def discretize(array, round_tolerance=0.3):
    """ Rounds each element of the array to the nearest integer
        and warns if any values are further than the tolerance from an integer.

        Parameters
        ----------
        array           : input array
        round_tolerance : float; maximum deviation from nearest integer before warning

        Returns
        -------
        rounded_array : output array """
    
    array = np.asarray(array)
    
    rounded_real = np.round(array.real)
    rounded_imag = np.round(array.imag)
    rounded_array = rounded_real + 1j * rounded_imag
    
    real_deviation = np.abs(array.real - rounded_real)
    imag_deviation = np.abs(array.imag - rounded_imag)
    
    # Combine mask: true where either part exceeds tolerance
    mask = (real_deviation > round_tolerance) | (imag_deviation > round_tolerance)
    if np.any(mask):
        print("Warning: Some elements are too far from integers to be safely rounded.\n")
        indices = np.argwhere(mask)
        for idx in indices:
            idx_tuple = tuple(idx)
            val = array[idx_tuple]
            rdev = real_deviation[idx_tuple]
            idev = imag_deviation[idx_tuple]
            print(f"Index {idx_tuple}: value = {val}, deviation = (real: {rdev:.3g}, imag: {idev:.3g})")
    print()
    
    # Return real if imaginary part is zero
    if np.all(rounded_imag == 0): return rounded_real
    else:                         return rounded_array

def _make_variable(sys, var_set):
    """ Creates and returns a variable.
        
        var_set
        -------
        'custom'   : prompted entry
        'standard' : default for most uses
        'debug'    : minimum for debugging other things
        list       : manual entry """
    
    if var_set == 'custom':
        lower    = float(input(f"{'variable lower bound (N * λ_crit)':<35}: "))
        upper    = float(input(f"{'variable upper bound (N * λ_crit)':<35}: "))
        samples  = int(input(f"{'number of trials':<35}: "))
        return np.linspace(lower*sys.crit, upper*sys.crit, samples)
    
    elif var_set == 'standard':
        return np.linspace(0.01, 2*sys.crit, 101)
    
    elif var_set == 'debug': 
        return np.linspace(0, 0.1, 2)
    
    elif type(var_set) == list:
        return np.linspace(var_set[0]*sys.crit, var_set[1]*sys.crit, var_set[2])

def _find_eigenstates(sys):
    """ Calculates eigenvalues and eigenvectors for the given matrix.
        For some reason, eigh provides the same eigenvectors as QuTiP, but eig does not.
        (Then why am I using it?)

        Returns
        -------
        states : list of arrays; sets the standard representation """
    
    eigenvalues, eigenvectors = [], []
    for i in tqdm(range(len(sys.H_list)), desc=f"{'finding eigenstates':<35}"):
        eigenvalue, eigenvector = np.linalg.eig(sys.H_list[i])
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
    sys.states_backup = [np.array(eigenvalues), np.array(eigenvectors)] 
    return sys.states_backup

def _select_states(sys, selection):
    """ Custructs sets of states from a set of eigenstates. See System.select() for details.
     
        Returns
        -------
        states  : 3D array; standard representation
        numbers : 3D array; standard representation= """
    
    # Select states manually
    if type(selection) == list:
    
        # Consolidate lists
        if type(selection[0]) == list:
            selection_cache = []
            
            # First parity; nothing interesting here
            for i in range(len(selection[0])):
                selection_cache.append(selection[0][i])
            
            # Second parity; shifts indices by half of the number of total states
            for i in range(len(selection[1])):
                selection_cache.append(int(len(sys.states[0][0])/2)+selection[1][i])
            selection = selection_cache
        
        # Update and return states
        states  = [sys.states[0][:, selection], sys.states[1][:, :, selection]]
        numbers = sys.numbers[:, selection]
        
        return states, numbers
    
    # Generate random state
    elif selection == 'random':
    
        # Initialize data container and randomization
        new_states = [[], []]
        seed  = np.random.randint(100)
        rng   = np.random.default_rng(seed)
        print(f"{'seed':<35}: {seed}")
        
        # Choose how many eigenstates to include
        num_eigenstates    = rng.integers(low=1, high=sys.states[0][0].shape[0])
        
        # Choose random eigenstates and weights for each eigenstate
        random_eigenstates = rng.integers(low=0, high=sys.states[0][0].shape[0], size=num_eigenstates)
        random_weights     = rng.uniform(0, 1, num_eigenstates)
        
        # Sort through trials
        for i in tqdm(range(sys.states[1].shape[0]), desc=f"{'constructing state':<35}"):
        
            # Construct state
            new_state = np.zeros_like(sys.states[1][0][:,0])
            for j in range(num_eigenstates):
                new_state += random_weights[j] * sys.states[1][i][:,random_eigenstates[j]]

            # Normalize and recast as column vector
            new_state = rtc(new_state / np.linalg.norm(new_state))

            # Calculate energy
            new_energy = expectation(sys.H_list[i], new_state)

            # Append to data container
            new_states[0].append([new_energy])
            new_states[1].append(new_state)

        new_states = [np.array(new_states[0]), np.array(new_states[1])]
        return new_states, None
    
    # Construct superposition
    elif selection == 'super':
        new_states = [[], []]
        for i in range(sys.states[1].shape[0]):
            new_states[0].append([0])
            new_states[1].append(np.zeros_like(sys.states[1][0][:,0]))
            for j in range(sys.states[1].shape[2]):
                new_states[0][-1] += sys.states[0][i][j]
                new_states[1][-1] += sys.states[1][i][:,j]
            new_states[1][-1] = rtc(new_states[1][-1] / np.linalg.norm(new_states[1][-1]))
        new_states = [np.array(new_states[0]), np.array(new_states[1])]
        return new_states, None
    
    # Select negative parity
    elif selection == 'parity':
        new_states = [[], []]
        
        # Sort through trials
        for i in tqdm(range(sys.states[1].shape[0]), desc=f"{'constructing state':<35}"):
            new_states[0].append([])
            new_states[1].append([])
            
            # Cycle through states of negative parity
            for j in range((sys.states[1][i].shape[1] - int(len(sys.states[0][0])/2))//2):
        
                new_states[0][-1].append(sys.states[0][i][j])
                new_states[1][-1].append(rtc(sys.states[1][i][:,j]))
        
        new_states = [np.array(new_states[0]), np.array(new_states[1])]
        return new_states, None
    
    # Select lower half of energy eigenstates
    elif selection == 'half':
        selection_cache = []
        num_states      = sys.states[0].shape[1]
        parity_split    = sys.states[0].shape[1] // 2
        
        # Generate indices
        for i in range(num_states // 4):
            selection_cache.append(i)
            selection_cache.append(i + parity_split)
        
        # Update and return states
        states  = [sys.states[0][:, selection_cache], sys.states[1][:, :, selection_cache]]
        numbers = sys.numbers[:, selection_cache]
        
        return states, numbers
        
    # Select lower quarter of energy eigenstates
    elif selection == 'quarter':
        selection_cache = []
        num_states      = sys.states[0].shape[1]
        parity_split    = sys.states[0].shape[1] // 2
        
        # Generate indices
        for i in range(num_states // 8):
            selection_cache.append(i)
            selection_cache.append(i + parity_split)
        
        # Update and return states
        states  = [sys.states[0][:, selection_cache], sys.states[1][:, :, selection_cache]]
        numbers = sys.numbers[:, selection_cache]
        
        return states, numbers
    
    # Project a state onto each eigenbasis?
    # I don't know what this does, and I don't know if it works
    else:
        new_states = [[], []]

        # Sort through trials
        for i in range(sys.states[1].shape[0]):
            
            # Prepare with zeros
            new_states[0].append([0])
            new_states[1].append(np.zeros_like(sys.states[1][0][:,0]))
            
            # Sort through states
            for j in range(sys.states[1].shape[1]):
                
                # Calculate projection
                c = np.vdot(sys.state, sys.states[1][i][:,j])
                
                new_states[0][-1] += c * sys.states[0][i][j]
                new_states[1][-1] += c * sys.states[1][i][:,j]
            new_states[1][-1] = rtc(new_states[1][-1] / np.linalg.norm(new_states[1][-1]))
        new_states = [np.array(new_states[0]), np.array(new_states[1])]
        return new_states, None

def _calculate_quantum_numbers(sys, sort_1=None, sort_2=None):
    """ Find quantum numbers for each eigenstate at λ=0, assuming H has been constructed.
    
        Parameters
        ----------
        sort_1 : string in sort_dict; sets first quantum number to sort
        sort_2 : string in ['P', 'n', 'J_z', 'E']; sets second quantum number to sort
    
        Returns
        ------
        expectation_list : 2D array; contains n and m_J for each state """
    
    # Default sorting parameters
    sort_dict = {'P':   sys.P,           # parity
                 'η':   sys.η,           # excitation number
                 'E':   None,            # energy
                 'n':   sys.a_dag@sys.a, # occupation
                 'J_z': sys.J_z}         # spin polarization
    
    # Cycle through each λ
    expectation_list = []
    precision = 5
    for i in tqdm(range(len(sys.states[1])), desc=f"{'calculating numbers':<35}"):
        expectations_rounded = []
        state = lambda j: rtc(sys.states[1][i][:,j])
        num_trials = sys.states[1][0].shape[0]
        
        # Calculate all quantum numbers (|P, n, J_z, E⟩)
        if sort_1 == None:
            P_expectations   = [expectation(sys.P,              state(j)) for j in range(num_trials)]
            n_expectations   = [expectation(sys.a_dag @ sys.a, state(j)) for j in range(num_trials)]
            J_z_expectations = [expectation(sys.J_z,            state(j)) for j in range(num_trials)]
            E_expectations   = states[0][i]

            for k in range(len(states[0][i])):
                expectations_rounded.append([
                    round(P_expectations[k],   precision),
                    round(n_expectations[k],   precision),
                    round(J_z_expectations[k], precision),
                    round(E_expectations[k],   precision)])
            expectation_list.append(np.array(expectations_rounded))
        
        # Calculate one or two quantum numbers
        else:

            # Calculate one quantum number (|sort_1⟩)
            if sort_2 == None:
                
                # Avoid recalculating energy values
                if sort_1 == 'E':
                    for k in range(num_trials):
                        expectations_rounded.append([round(sys.states[0][i][k]), precision])
                    expectation_list.append(np.array(expectations_rounded))
                
                # Calculate quantum number
                else:
                    expectations_cache = [expectation(sort_dict[sort_1], state(j)) for j in range(num_trials)]
                    for k in range(num_trials):
                        expectations_rounded.append([round(expectations_cache[k]), precision])
                    expectation_list.append(np.array(expectations_rounded))
            
            # Calculate two quantum numbers (|sort_1, sort_2⟩)
            else:
                
                # Calculate first number
                operator = sort_dict[sort_1]
                if sort_1 == 'E':   cache_1 = sys.states[0][i]
                elif sort_1 == 'P': cache_1 = [round(expectation(operator, state(j)), precision) for j in range(num_trials)]
                elif sort_1 == 'η': cache_1 = [round(expectation(operator, state(j)), precision) for j in range(num_trials)]
                else:               cache_1 = [expectation(operator, state(j))                   for j in range(num_trials)]
                
                # Calculate second number
                if sort_2 == 'E': cache_2 = sys.states[0][i]
                else:             cache_2 = [expectation(sort_dict[sort_2], state(j)) for j in range(num_trials)]
                    
                for k in range(len(sys.states[0][i])):
                    expectations_rounded.append([
                        round(np.real(cache_1[k]), precision),
                        round(np.real(cache_2[k]), precision)])
                expectation_list.append(np.array(expectations_rounded))
    
    return np.array(expectation_list)

########################################################################################################################################################
# Classes
class System:

    def __init__(self, ω, ω0, field_dim, N, ℏ=1, m=1, spin=1/2, indiv=False, mod='Dicke', var_set='custom'):
        """ Initializes operators and parameters.
        
            Parameters
            ----------
            ω         : float; field frequency
            ω0        : float; atomic frequency
            field_dim : int; number of field modes
            N         : int; number of particles
            
            ℏ         : float; Planck's constant
            m         : float; mass
            spin      : int or half-int; atomic spin
            indiv     : bool; collective spin or individual spins
            
            mod       : str; preset Hamiltonian; see _Hamiltonian() for details
            var_set   : str or list; see _make_variable() for details """
        
        # Initialize parameters
        self.ω         = ω
        self.ω0        = ω0
        self.field_dim = field_dim
        self.N         = N
        self.crit      = (ω * ω0)**(1/2)
        self.ℏ         = ℏ
        self.m         = m
        self.spin      = spin
        self.J         = N/2
        self.mod       = mod
        
        if (mod == 'Ising') or (indiv == True):
            self.indiv    = True
            self.spin_dim = 2**N
        else:
            self.indiv    = False
            self.spin_dim = int(N+1)
        
        # Initialize things to be assigned later
        self.states         = None # selected eigenstates
        self.states_backup  = None # complete set of eigenstates
        self.numbers        = None # selected eigenvalues
        self.numbers_backup = None # complete set of eigenvalues
        self.plot_lists     = None # last plotted results
        self.J_x_list       = None # operators for each spin in tensor product space; only used for indiv=True
        self.J_y_list       = None
        self.J_z_list       = None
        self.J_p_list       = None
        self.J_m_list       = None

        # Create operators in subspaces
        J_spin_dict      = _create_J_operators(self)
        a_field_dict     = _create_a_operators(self)
        self.J_x_spin    = tolerance(J_spin_dict['J_x'])
        self.J_y_spin    = tolerance(J_spin_dict['J_y'])
        self.J_z_spin    = tolerance(J_spin_dict['J_z'])
        self.J_m_spin    = tolerance(J_spin_dict['J_m'])
        self.J_p_spin    = tolerance(J_spin_dict['J_p'])
        self.J2_spin     = tolerance((self.J_x_spin@self.J_x_spin)+(self.J_y_spin@self.J_y_spin)+(self.J_z_spin@self.J_z_spin))
        self.a_field     = tolerance(a_field_dict['a'])
        self.a_dag_field = tolerance(a_field_dict['a_dag'])
        del J_spin_dict, a_field_dict
        
        # Create operators in tensor product space
        J_a_dict   = _compute_tensor_products(self)
        self.J_x   = tolerance(J_a_dict['J_x'])     # spin x operator
        self.J_y   = tolerance(J_a_dict['J_y'])     # spin y operator
        self.J_z   = tolerance(J_a_dict['J_z'])     # spin z operator
        self.J_m   = tolerance(J_a_dict['J_m'])     # spin annihilation operator
        self.J_p   = tolerance(J_a_dict['J_p'])     # spin creation operator
        self.J2    = tolerance((self.J_x@self.J_x)+(self.J_y@self.J_y)+(self.J_z+self.J_z))
        self.a     = tolerance(J_a_dict['a'])       # field annihilation operator
        self.a_dag = tolerance(J_a_dict['a_dag'])   # field creation operator
        η_P_dict   = _create_excitation_operators(self)
        self.η     = tolerance(η_P_dict['η'])       # parity operator
        self.P     = tolerance(η_P_dict['P'])       # parity operator
        del J_a_dict, η_P_dict
        
        # Set variable and find states
        self.var    = _make_variable(self, var_set)       # variable array
        self.H_list = self._Hamiltonian(mod, indiv) # Hamiltonian matrix for each variable value
        self.states = _find_eigenstates(self)
        self.sort()

    def _Hamiltonian(self, mod, indiv):
        """ Creates a Hamiltonian for each variable in the set, then finds eigenstates.
            Set self.indiv=True to use non-collective spin.
        
            mod 
            ---
            'Dicke' : Dicke model; co-rotating and counter-rotating interactions between field and spin
            'pert'  : Dicke model with a field perturbation in position quadrature
            'SOC'   : Dicke model under the unitary transformation e^(iθJ_y) with momentum quadrature instead of position
            'TC'    : Tavis-Cummings model; co-rotating interaction between field and spin
            'Ising' : Ising model; nearest neighbor spin-spin interaction without a field
            
            indiv
            -----
            True    : full spin space (atypical)
            False   : collective spin space (typical) """
        
        # Choose a Hamiltonian
        if self.mod == 'Dicke':
            H_field = self.ℏ * self.ω  * (self.a_dag @ self.a)
            H_atom  = self.ℏ * self.ω0 * self.J_z
            H_int   = self.ℏ / np.sqrt(self.N) * (self.a_dag + self.a) @ self.J_x
            H       = lambda λ: H_field + H_atom + λ*H_int
        
        elif self.mod == 'pert':
            H_field = self.ℏ * self.ω  * (self.a_dag @ self.a)
            H_atom  = self.ℏ * self.ω0 * self.J_z
            H_int   = self.ℏ / np.sqrt(self.N) * (self.a_dag + self.a) @ self.J_x
            H_pert  = 10**(-10) * (self.a_dag + self.a)
            H       = lambda λ: H_field + H_atom + λ*H_int + H_pert
        
        elif self.mod == 'SOC':
            δ        = float(input(f"{'detuning':<35}: "))
            H_field  = self.ℏ * self.ω * (self.a_dag @ self.a)
            H_atom_x = self.ℏ * self.ω0 * self.J_x
            H_atom_z = self.ℏ * δ/2 * self.J_z
            H_int    = self.ℏ / np.sqrt(self.N) * 1j * (self.a_dag - self.a) @ self.J_z
            H        = lambda λ: H_field + H_atom_x + H_atom_z + λ*H_int
        
        elif self.mod == 'TC':
            H_field = self.ℏ * self.ω  * (self.a_dag @ self.a)
            H_atom  = self.ℏ * self.ω0 * self.J_z
            H_int   = self.ℏ / (2 * np.sqrt(self.N)) * (self.a_dag @ self.J_m + self.a @ self.J_p)
            H       = lambda λ: H_field + H_atom + λ*H_int
        
        elif self.mod == 'Ising':
            H_int = np.zeros_like(self.J_z)
            for i in range(self.N):
                for j in range(self.N):
                    if (j == i-1) or (j == i+1):
                        H_int += self.J_z_list[i] @ self.J_z_list[j]
            H_int  = self.ℏ * self.ω0 * H_int
            H_atom = self.ℏ * self.ω0 * self.J_z
            H      = lambda λ: H_atom + λ*H_int
        
        else:
            print(f"{'Error! Unsupported mod':<35}: {self.mod}\n")
            return
        
        # Calculate and return each Hamiltonian in the parameter space
        H_list = []
        for i in range(len(self.var)):
            H_list.append(H(self.var[i]))
        return np.array(H_list)

    def sort(self, sort_1='P', sort_2='E'):
        """ Finds quantum numbers for each eigenstate at λ=0, then sorts the eigenvectors accordingly.
        
            Parameters
            ----------
            sort_1 : string in ['E', 'P', 'J_z', 'n']
            sort_2 : string in ['E', 'P', 'J_z', 'n'] """
        
        # Initialize temporary data containers
        sorted_states_0     = []
        sorted_states_1     = []
        sorted_expectations = []

        # Find numbers for each state
        expectation_list = _calculate_quantum_numbers(self, sort_1, sort_2)

        # Loop over each set of states
        for i in range(len(expectation_list)):
            row = expectation_list[i]
            
            # Sort by secondary eigenvalue parameter first (if provided)
            if sort_2: sorted_indices = np.argsort(row[:, 1], kind='stable')
            else:      sorted_indices = np.arange(len(row))
            
            # Sort by the primary eigenvalue parameter while preserving secondary order
            sorted_indices = sorted_indices[np.argsort(row[sorted_indices, 0], kind='stable')]
            sorted_row = row[sorted_indices]
            sorted_states_0.append(np.array(self.states[0][i])[sorted_indices]) 
            sorted_states_1.append(self.states[1][i][:, sorted_indices])
            sorted_expectations.append(sorted_row)
        
        # Convert and export
        sorted_states_0 = np.array(sorted_states_0)
        sorted_states_1 = np.array(sorted_states_1)
        self.states     = [sorted_states_0, sorted_states_1]
        self.numbers    = np.array(sorted_expectations)

    def select(self, selection):
        """ Manages the states used for plotting and analysis.
            
            selection
            --------
            list         : by index
            nested lists : by index for each parity
            'random'     : random state as a linear combination
            'super'      : single superposition of all current states
            'parity'     : all states with negative parity
            'half'       : lowest half of all current states 
            'quarter'    : lowest quarter of all current states
            'backup'     : utility; saves current states to states_backup
            'restore'    : utility; replaces current states with states_backup """

        # Create a backup
        if (selection == 'backup') or (self.states_backup == None):
            self.states_backup  = self.states
            self.numbers_backup = self.numbers
        
        # Restore a backup
        if (selection == 'restore') or (self.states[0].shape != self.states_backup[0].shape):
            if selection != 'super':
                self.states  = self.states_backup
                self.numbers = self.numbers_backup
                self.sort()

        # Select states
        if selection not in ['backup', 'restore']:
            self.states, self.numbers = _select_states(self, selection)

    def restore(self):
        """ A shortcut for sys.select('restore'). Stupid but useful. """
        self.select('restore')

    def plot(self, selection='spectrum'):
        """ Generates common plots by keyword.

            Options
            -------
            spectrum   : energy expectation; any number of states
            occupation : cavity occupation and polarization; any number of states
            squeezing  : field squeezing; fewer than five states
            entropy    : field-spin entanglement; fewer than five states
            all        : energy and squeezing, occupation and polarization, and entropy; fewer than five states
            last       : whatever was previously plotted
            spins      : polarization for collective and individual spins; fewer than five states
            Chebyshev  : time evolution; fewer than five states
            Lindblad   : time evolution; fewer than five states
            spacing    : binned differences of adjacent energy eigenvalues """
    
        self.print('parameters')
        self.print('numbers')
    
        # Spectrum
        if selection == 'spectrum':
            plot_list = _spectrum(self)
        
        # Occupation
        elif selection == 'occupation':
            plot_list = _occupation(self)
        
        # Squeezing
        elif selection == 'squeezing':
            ΔJ_x      = uncertainty(self.J_x, self.states)
            ΔJ_y      = uncertainty(self.J_y, self.states)
            ΔJ_z      = uncertainty(self.J_z, self.states)
            J_x_exp   = expectation(self.J_x, self.states)
            J_x_exp   = tolerance(J_x_exp)
            J_y_exp   = expectation(self.J_y, self.states)
            J_z_exp   = expectation(self.J_z, self.states)
            product_1 = ΔJ_x * ΔJ_y
            product_2 = ΔJ_y * ΔJ_z
            product_3 = ΔJ_x * ΔJ_z
            ζ         = _squeezing(self)
            plot_list = [
                [(f"", f"$ζ^2$"),      (self.var, ζ),         (0, 1), ('plot')],
                [(f"", f"$⟨J_x⟩$"),     (self.var, J_x_exp),   (1, 0), ('plot')],
                [(f"", f"$⟨J_y⟩$"),     (self.var, J_y_exp),   (1, 1), ('plot')],
                [(f"", f"$⟨J_z⟩$"),     (self.var, J_z_exp),   (1, 2), ('plot')],
                [(f"", f"$ΔJ_x$"),     (self.var, ΔJ_x),      (2, 0), ('plot')],
                [(f"", f"$ΔJ_y$"),     (self.var, ΔJ_y),      (2, 1), ('plot')],
                [(f"", f"$ΔJ_z$"),     (self.var, ΔJ_z),      (2, 2), ('plot')],
                [(f"", f"$ΔJ_xΔJ_y$"), (self.var, product_1), (3, 0), ('plot')],
                [(f"", f"$ΔJ_yΔJ_z$"), (self.var, product_2), (3, 1), ('plot')],
                [(f"", f"$ΔJ_xΔJ_z$"), (self.var, product_3), (3, 2), ('plot')]]
        
        # Entropy
        elif selection == 'entropy':
            plot_list = _entropy(self)
        
        # All
        elif selection == 'all':
            plot_list_E       = _spectrum(self)
            plot_list_E.append([(f"λ", f"$ζ^2$"), (self.var, _squeezing(self)), (0, 1), ('plot')])
            plot_list_n_J     = _occupation(self)
            plot_list_entropy = _entropy(self)
            
            self.plot_lists = [[plot_list_E, plot_list_n_J, plot_list_entropy], self.numbers]
            
            plot_handling(plot_list_E,       self.numbers, no_show=True)
            plot_handling(plot_list_n_J,     self.numbers, no_show=True)
            plot_handling(plot_list_entropy, self.numbers, no_show=True)
            plt.show()
        
        # Last
        elif selection == 'last':
            for i in range(len(self.plot_lists[0])):
                plot_handling(self.plot_lists[0][i], self.plot_lists[1], no_show=True)
            plt.show()
        
        # Spins
        elif selection == 'spins':
            if not self.indiv:
                print("Error! Reconstruct Hamiltonian with indiv set to True for this feature.")
                return
            else:
                expected  = expectation(self.J_z, self.states)
                plot_list = [[(f"$λ$", f"$⟨J_z⟩$"), (self.var, expected), (0, 0), ('plot')]]
                J_z_cache = []
                for i in range(self.N):
                    expected = expectation(self.J_z_list[i], self.states)
                    plot_list.append([(f"$λ$", f"$⟨J_{i+1}⟩$"), (self.var, expected), (1, i), ('plot')])
        
        # Chebyshev evolution
        elif selection == 'Chebyshev':
            custom = bool(input(f"{'custom [True, False]'}: "))
            plot_list = _Chebyshev(self, custom)
            self.plot_lists = [[plot_list], self.numbers]
            plot_handling(plot_list, plot_mode='3D', sys=self)
        
        # Lindblad evolution
        elif selection == 'Lindblad':
            custom = bool(input(f"{'custom [True, False]'}: "))
            plot_list = _Lindbladian(self, custom)
            self.plot_lists = [[plot_list], self.numbers]
            plot_handling(plot_list, plot_mode='3D', sys=self)
        
        # Energy spacing histogram
        elif selection == 'spacing':
            _E_spacing(self)
            return
        
        else: print('!! Try a different keyword !!\n')

        # Plot calculated values
        if selection not in ['all', 'last', 'Chebyshev', 'Lindblad']:
            self.plot_lists = [[plot_list], self.numbers] # save for later
            plot_handling(plot_list, self.numbers)

    @classmethod
    def load(cls, filename='cache'):
        """ Loads states. Use sys=System.load(filename) if a System object has not been previously constructed. """
        with open('Dicke_' + filename + '.pkl', 'rb') as file:
            system_instance = pickle.load(file)
        return system_instance

    def save(self, filename='cache'):
        """ Save states with sys.save('filename'). """
        filename = 'Dicke_' + filename + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def _update(self):
        """ Run this every time a parameter is changed. """
        self.__init__(self.ω, self.ω0, self.field_dim, self.N, self.ℏ, self.m, self.spin)

    def print(self, data='parameters'):
        if data == 'parameters':
            print(f"\n-------------------------------------------------------------\n"
                  f"{'model':<35}: {self.mod:<10}\n"
                  f"{'field frequency':<35}: {round(self.ω, 2):<4}\n"
                  f"{'atomic frequency':<35}: {round(self.ω0, 2):<4}\n"
                  f"{'critical coupling':<35}: {round(self.crit, 2):<4}\n"
                  f"{'number of modes':<35}: {self.field_dim:<4}\n"
                  f"{'number of particles':<35}: {self.N:<4}\n"
                  f"-------------------------------------------------------------\n")
        
        elif data == 'numbers':
            try:
                print(f"\n-------------------------------------------------------------")
                print(f"|P, E)\t\t |P, E)\t\t |P, E)\t\t |P, E)")
                
                if len(self.numbers[0]) <= 4:
                    cache = f""
                    for i in range(len(self.numbers[0])):
                        cache += f"|{round(self.numbers[0][i][0])}, {round(self.numbers[0][i][1], 2)})\t "
                    print(cache)

                else:
                    for i in range(len(self.numbers[0])//4):
                        print(f"|{round(self.numbers[0][4*i][0])}, {round(self.numbers[0][4*i][1], 2)})"
                            f"\t |{round(self.numbers[0][4*i+1][0])}, {round(self.numbers[0][4*i+1][1], 2)})"
                            f"\t |{round(self.numbers[0][4*i+2][0])}, {round(self.numbers[0][4*i+2][1], 2)})"
                            f"\t |{round(self.numbers[0][4*i+3][0])}, {round(self.numbers[0][4*i+3][1], 2)})")
                    if (len(self.numbers[0])/4 - len(self.numbers[0])//4) != 0:
                        for i in range(len(self.numbers[0]) - len(self.numbers[0]//4)):
                            print(f"|{self.numbers[0][i][0]}, {round(self.numbers[0][i][1], 2)}⟩\t")
                print(f"-------------------------------------------------------------\n")
            except:
                pass
        elif data == 'all numbers':
            print(self.numbers)

########################################################################################################################################################
# Main
def main():
    examples(0)    

if __name__ == '__main__':
    main()

########################################################################################################################################################

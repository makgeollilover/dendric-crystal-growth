import numpy as np
import matplotlib.pyplot as plt

# Set domain size
L_x, L_y = 50, 50  # Length in x and y direction
N_x, N_y = 51, 51  # Number of grid points in x and y direction

# Define grid
x, y = np.linspace(0, L_x, N_x), np.linspace(0, L_y, N_y)
X, Y = np.meshgrid(x, y)

# Grid spacing
h = x[1] - x[0]
Klist = [0.8,1.0,2.0]
for K in Klist:
    # Set initial conditions
    p_0 = 0  # 0 = liquid, 1 = solid
    T_0 = 298.15  # Room temperature

    alpha = 0.01  # Thermal diffusivity
    tau = 0.003
    #K = 2  # Dimensionless latent heat
    m = 0.45

    # temperature grid starting from T_0
    T = np.full((N_x, N_y), T_0)


    p = np.full((N_x, N_y), p_0)  # Set p grid with initial condition

    ## Set a solid seed
    #seed_radius = 13  # Radius
    #seed_center = (L_x / 2, L_y / 2)  # Position
    #for i in range(N_x):  # Set p to solid in radius
    #    for j in range(N_y):
    #        if (x[i] - seed_center[0])**2 + (y[j] - seed_center[1])**2 < seed_radius**2:
    #            p[i, j] = 1

    # Set a ring-shaped seed
    seed_center = (L_x / 2, L_y / 2)  # Position
    r_inner = 8  # Inner radius of the ring
    r_outer = 9  # Outer radius of the ring

    for i in range(N_x):
        for j in range(N_y):
            r_squared = (x[i] - seed_center[0])**2 + (y[j] - seed_center[1])**2
            if r_inner**2 < r_squared < r_outer**2:
                p[i, j] = 1


    # Define time steps and number of steps
    d_t = 0.002  # Adjust the time step if needed
    n_steps = 1000


    # Plot initial conditions
    plt.figure()
    plt.imshow(p, extent=[0, L_x, 0, L_y], origin='lower', cmap='gray')
    plt.colorbar(label='Phase Field')
    plt.title(f'Initial Phase Field Distribution K = {K} m = {m}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"results\seed_initial_K{K}.png")
    plt.close()
    #plt.show()

    plt.figure()
    plt.imshow(T, extent=[0, L_x, 0, L_y], origin='lower', cmap='hot_r')
    plt.colorbar(label='Temperature (K)')
    plt.title(f'Initial Temperature Distribution K = {K}')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.savefig(f"results\heat_initial_K{K}m{m}.png")
    plt.close()
    #plt.show()


    for step in range(n_steps):
        p_new = p.copy()
        T_new = T.copy()

        # Update field equation
        for i in range(1, N_x-1):
            for j in range(1, N_y-1):
                # Phase field update (discretized)
                Te = 1 #characteristic cooling T
                m = 0.9/np.pi*np.arctan(10*(T[i,j]-Te)) #assure absolute value is smaller than 0.5
                
                laplacian_p = ((p[i+1, j]+ p[i, j-1]) + p[i, j+1] + p[i-1, j] - 4*p[i, j]) / h**2
                p_update = 1/tau*d_t * (laplacian_p - p[i, j] * (1 - p[i, j]) * (p[i, j] - 0.5 + m))
                p_new[i, j] = p[i, j] + p_update

        # Apply zero flux boundary conditions for phase field
        p_new[0, :] = p_new[1, :]
        p_new[-1, :] = p_new[-2, :]
        p_new[:, 0] = p_new[:, 1]
        p_new[:, -1] = p_new[:, -2]

            # Ensure phase field values remain between 0 and 1
        p_new = np.clip(p_new, 0, 1)
        #p_new = np.round(p_new)

        # Update temperature field
        for i in range(1, N_x - 1):
            for j in range(1, N_y - 1):
                laplacian_T = (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - 4*T[i, j]) / h**2
                T_new_ij_update = d_t * (laplacian_T + K * (p_new[i, j] - p[i, j]) / d_t)
                T_new[i, j] = T[i, j] + T_new_ij_update

        # Apply zero flux boundary conditions for temperature field
        T_new[0, :] = T_new[1, :]
        T_new[-1, :] = T_new[-2, :]
        T_new[:, 0] = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]

        # Update fields
        p = p_new.copy()
        T = T_new.copy()

        # Debugging: Print some values at certain steps
        if step == 200 or step == 1 or step == 0:# == 0:
            print(f'Step {step}: max(p)={np.max(p)}, min(p)={np.min(p)}, max(T)={np.max(T)}, min(T)={np.min(T)}')
            #print(T[9,9])
            plt.figure()
            plt.imshow(p, extent=[0, L_x, 0, L_y], origin='lower', cmap='gray')
            plt.colorbar(label='Phase Field')
            plt.title(f'Phase Field Distribution at step {step} for K = {K} m = {m}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(f"results\seed_step_{step}_K{K}m{m}.png")
            plt.close()
            #plt.show()

            plt.figure()
            plt.imshow(T, extent=[0, L_x, 0, L_y], origin='lower', cmap='hot_r')
            plt.colorbar(label='Temperature (K)')
            plt.title(f'Temperature Distribution at step {step} for K = {K} m = {m}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(f"results\heat_{step}_K{K}m{m}.png")
            plt.close()
            #plt.show()


    # Plot initial conditions
    print(f'Step {step}: max(p)={np.max(p)}, min(p)={np.min(p)}, max(T)={np.max(T)}, min(T)={np.min(T)}')
    plt.figure()
    plt.imshow(p, extent=[0, L_x, 0, L_y], origin='lower', cmap='gray')
    plt.colorbar(label='Phase Field')
    plt.title(f'Final Phase Field Distribution K = {K} m = {m}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"results\seed_step_{step}_K{K}m{m}.png")
    plt.close()
    #plt.show()

    plt.figure()
    plt.imshow(T, extent=[0, L_x, 0, L_y], origin='lower', cmap='hot_r')
    plt.colorbar(label='Temperature (K)')
    plt.title(f'Final Temperature Distribution K = {K} m = {m}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"results\heat_{step}_K{K}m{m}.png")
    plt.close()
    #plt.show()
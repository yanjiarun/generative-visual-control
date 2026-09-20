import numpy as np
import matplotlib.pyplot as plt

def compute_cassini_field(X, Y, a=1.5, b=0.9):
    """Compute vector field for Cassini oval: α = (x²+y²)² - 2a²(x²-y²) - b⁴ = 0"""
    # Define α function and its gradient
    alpha = (X**2 + Y**2)**2 - 2*a**2*(X**2 - Y**2) - b**4
    dalpha_dx = 4*X*(X**2 + Y**2) - 4*a**2*X
    dalpha_dy = 4*Y*(X**2 + Y**2) + 4*a**2*Y
    
    # Define potential function P = ½α² and its gradient
    P = 0.5 * alpha**2
    dP_dx = alpha * dalpha_dx
    dP_dy = alpha * dalpha_dy
    
    # Compute G and H functions (paper eq. 4)
    G = -(2/np.pi) * np.arctan(2*np.sqrt(P))
    H = np.sqrt(1 - G**2)
    
    # Compute normalized gradients
    norm_dP = np.sqrt(dP_dx**2 + dP_dy**2)
    norm_Rdalpha = np.sqrt(dalpha_dy**2 + dalpha_dx**2)

    # Avoid division by zero
    epsilon = 1e-8
    norm_dP = np.where(norm_dP < epsilon, epsilon, norm_dP)
    norm_Rdalpha = np.where(norm_Rdalpha < epsilon, epsilon, norm_Rdalpha)

    # Compute vector field components
    u_x = G * (dP_dx/norm_dP) + H * (-dalpha_dy/norm_Rdalpha)
    u_y = G * (dP_dy/norm_dP) + H * (dalpha_dx/norm_Rdalpha)

    # Normalize vector field with epsilon to avoid division by zero
    # epsilon = 1e-8
    # norm_u = np.sqrt(u_x**2 + u_y**2)
    # norm_u_safe = np.where(norm_u < epsilon, epsilon, norm_u)
    # u_x_norm = u_x / norm_u_safe
    # u_y_norm = u_y / norm_u_safe
    
    return u_x, u_y, alpha

def compute_cassini_field_with_gradients(X, Y, a=1.5, b=0.9):
    """Compute vector field with gradients for Cassini oval: α = (x²+y²)² - 2a²(x²-y²) - b⁴ = 0"""
    # Define α function and its gradient
    alpha = (X**2 + Y**2)**2 - 2*a**2*(X**2 - Y**2) - b**4
    dalpha_dx = 4*X*(X**2 + Y**2) - 4*a**2*X
    dalpha_dy = 4*Y*(X**2 + Y**2) + 4*a**2*Y
    
    # Define potential function P = ½α² and its gradient
    P = 0.5 * alpha**2
    dP_dx = alpha * dalpha_dx
    dP_dy = alpha * dalpha_dy
    
    # Compute G and H functions (paper eq. 4)
    G = -(2/np.pi) * np.arctan(2*np.sqrt(P))
    H = np.sqrt(1 - G**2)
    
    # Compute normalized gradients
    norm_dP = np.sqrt(dP_dx**2 + dP_dy**2)
    norm_Rdalpha = np.sqrt(dalpha_dy**2 + dalpha_dx**2)

    # Avoid division by zero
    epsilon = 1e-8
    norm_dP = np.where(norm_dP < epsilon, epsilon, norm_dP)
    norm_Rdalpha = np.where(norm_Rdalpha < epsilon, epsilon, norm_Rdalpha)

    # Compute vector field components
    u_x = G * (dP_dx/norm_dP) + H * (-dalpha_dy/norm_Rdalpha)
    u_y = G * (dP_dy/norm_dP) + H * (dalpha_dx/norm_Rdalpha)

    # Compute gradients of the vector field components
    dG_dP = -(4/np.pi) / (1 + 4*P) * (0.5/np.sqrt(np.abs(P) + epsilon))
    dG_dx = dG_dP * dP_dx
    dG_dy = dG_dP * dP_dy
        
    dH_dG = -G / (H + epsilon)
    dH_dx = dH_dG * dG_dx
    dH_dy = dH_dG * dG_dy
        
    # Compute second derivatives of α
    d2alpha_dx2 = 4*(X**2 + Y**2) + 8*X**2 - 4*a**2
    d2alpha_dy2 = 4*(X**2 + Y**2) + 8*Y**2 + 4*a**2
    d2alpha_dxdy = 8*X*Y
        
    # Compute second derivatives of P
    d2P_dx2 = dalpha_dx**2 + alpha * d2alpha_dx2
    d2P_dy2 = dalpha_dy**2 + alpha * d2alpha_dy2
    d2P_dxdy = dalpha_dx * dalpha_dy + alpha * d2alpha_dxdy
        
    # Compute gradients of norm_dP
    dnorm_dP_dx = (dP_dx*d2P_dx2 + dP_dy*d2P_dxdy) / norm_dP
    dnorm_dP_dy = (dP_dx*d2P_dxdy + dP_dy*d2P_dy2) / norm_dP
        
    # Compute gradients of unit vector e1 = (dP_dx/norm_dP, dP_dy/norm_dP)
    de1x_dx = (d2P_dx2*norm_dP - dP_dx*dnorm_dP_dx) / (norm_dP**2)
    de1x_dy = (d2P_dxdy*norm_dP - dP_dx*dnorm_dP_dy) / (norm_dP**2)
    de1y_dx = (d2P_dxdy*norm_dP - dP_dy*dnorm_dP_dx) / (norm_dP**2)
    de1y_dy = (d2P_dy2*norm_dP - dP_dy*dnorm_dP_dy) / (norm_dP**2)
        
    # Compute gradients of norm_Rdalpha
    dnorm_Rdalpha_dx = (dalpha_dy*d2alpha_dxdy + dalpha_dx*d2alpha_dx2) / norm_Rdalpha
    dnorm_Rdalpha_dy = (dalpha_dy*d2alpha_dy2 + dalpha_dx*d2alpha_dxdy) / norm_Rdalpha
        
    # Compute gradients of unit vector e2 = (-dalpha_dy/norm_Rdalpha, dalpha_dx/norm_Rdalpha)
    de2x_dx = (-d2alpha_dxdy*norm_Rdalpha + dalpha_dy*dnorm_Rdalpha_dx) / (norm_Rdalpha**2)
    de2x_dy = (-d2alpha_dy2*norm_Rdalpha + dalpha_dy*dnorm_Rdalpha_dy) / (norm_Rdalpha**2)
    de2y_dx = (d2alpha_dx2*norm_Rdalpha - dalpha_dx*dnorm_Rdalpha_dx) / (norm_Rdalpha**2)
    de2y_dy = (d2alpha_dxdy*norm_Rdalpha - dalpha_dx*dnorm_Rdalpha_dy) / (norm_Rdalpha**2)
        
    # Compute gradients of vector field components
    du_x_dx = dG_dx*(dP_dx/norm_dP) + G*de1x_dx + dH_dx*(-dalpha_dy/norm_Rdalpha) + H*de2x_dx
    du_x_dy = dG_dy*(dP_dx/norm_dP) + G*de1x_dy + dH_dy*(-dalpha_dy/norm_Rdalpha) + H*de2x_dy
    du_y_dx = dG_dx*(dP_dy/norm_dP) + G*de1y_dx + dH_dx*(dalpha_dx/norm_Rdalpha) + H*de2y_dx
    du_y_dy = dG_dy*(dP_dy/norm_dP) + G*de1y_dy + dH_dy*(dalpha_dx/norm_Rdalpha) + H*de2y_dy
        
    delta_ux = np.array([du_x_dx, du_x_dy])
    delta_uy = np.array([du_y_dx, du_y_dy])
    
    return u_x, u_y, alpha, delta_ux, delta_uy

def plot_cassini_field(X, Y, u_x, u_y, alpha, a=1.5, b=0.9, sample_rate=4, scale=15):
    """Plot vector field with quiver and streamlines"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
    
    # Left: Quiver plot
    ax1.contour(X, Y, alpha, levels=[0], colors='red', linewidths=2)
    ax1.quiver(X[::sample_rate, ::sample_rate], Y[::sample_rate, ::sample_rate], 
               u_x[::sample_rate, ::sample_rate], u_y[::sample_rate, ::sample_rate], 
               color='blue', alpha=0.5, scale=scale)
    ax1.set_title(f'Cassini Oval: a={a}, b={b}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim([-2.8, 2.8])
    ax1.set_ylim([-1.4, 1.4])
    ax1.grid(True, alpha=0.3)
    
    # Right: Streamlines
    ax2.streamplot(X, Y, u_x, u_y, color='green', density=2)
    ax2.contour(X, Y, alpha, levels=[0], colors='red', linewidths=2)
    ax2.set_title('Streamlines')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim([-2.8, 2.8])
    ax2.set_ylim([-1.4, 1.4])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_cassini_experiment(a=1.5, b=0.9):    # 0.8 1.0
    """Main function to create and visualize Cassini oval vector field"""
    n_points = 70
    x = np.linspace(-2.8, 2.8, n_points+1)
    y = np.linspace(-1.4, 1.4, int(n_points * 1/2)+1)  # n_points * (y_span/x_span)
    X, Y = np.meshgrid(x, y)
    
    u_x, u_y, alpha = compute_cassini_field(X, Y, a, b)
    plot_cassini_field(X, Y, u_x, u_y, alpha, a, b, sample_rate=1, scale=50)

if __name__ == '__main__':
    print("Creating Cassini oval vector field...")
    create_cassini_experiment()
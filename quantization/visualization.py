import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import matplotlib


def dot_figure(reference_distribution_P_norm_final, curDistributionEdge_P, xlim, ymax, minKL_threshold, ifold, iLayer,
               fig_type="origin"):
    """
    Generate and save a density-enhanced scatter plot for distribution analysis.

    Args:
        reference_distribution_P_norm_final: Array of normalized reference distribution values
        curDistributionEdge_P: Array of current distribution edge points
        xlim: Upper limit for x-axis
        ymax: Reference maximum y value (used as a safety upper bound)
        minKL_threshold: KL divergence threshold value
        ifold: Current fold index (0-based)
        iLayer: Current layer index (0-based)
        fig_type: Figure type identifier ("origin" or "final")
    """
    # Set a larger font size for this plot (double the previous size)
    plt.rcParams.update({
        'font.size': 48,  # Base font size
        'axes.titlesize': 64,  # Title font
        'axes.labelsize': 56,  # Axis labels
        'xtick.labelsize': 48,  # x-axis tick labels
        'ytick.labelsize': 48,  # y-axis tick labels
        'legend.fontsize': 48,  # Legend
    })

    # Set a larger figure size to accommodate larger fonts
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(24, 18), dpi=120)  # Increase canvas size

    # Data preprocessing
    eps = 1e-10
    valid_ref = np.maximum(reference_distribution_P_norm_final, eps)

    # Remove infinity and NaN values
    valid_mask = np.isfinite(curDistributionEdge_P[:-1]) & np.isfinite(np.log10(valid_ref))
    x_data = curDistributionEdge_P[:-1][valid_mask]
    y_data = valid_ref[valid_mask]

    # Ensure data is not empty
    if len(x_data) < 2:
        print(f"Warning: Not enough valid data points for Layer {iLayer + 1}")
        return

    # Calculate y-axis range
    y_min = 1e-7  # Keep a fixed minimum value

    # Use the maximum value of the data and add an appropriate margin
    y_max_data = np.max(y_data)
    log_y_max = np.log10(y_max_data)
    log_margin = 0.1 * abs(log_y_max)
    y_max = max(10 ** (log_y_max + log_margin), ymax)  # Use ymax as an upper bound

    # Prepare data for density estimation
    xy = np.vstack([x_data, np.log10(y_data)])

    # Optimize bandwidth selection
    n_points = xy.shape[1]
    if n_points > 100:
        bw_method = 'scott'
    else:
        bw_method = np.clip(0.1 * np.sqrt(100 / n_points), 0.05, 0.5)

    try:
        kde = gaussian_kde(xy, bw_method=bw_method)
        z = kde(xy)

        # Enhance density visualization
        z = np.power(z, 0.2)
        z = (z - z.min()) / (z.max() - z.min())
        z = np.power(z, 0.5)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Warning: KDE estimation failed for Layer {iLayer + 1}, Fold {ifold + 1}: {str(e)}")
        z = np.ones_like(x_data)

    # Color scheme
    colors = [
        (0.95, 0.95, 0.95),
        (0.85, 0.90, 0.95),
        (0.65, 0.75, 0.85),
        (0.45, 0.55, 0.65)
    ]
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue', colors)

    # Create the main scatter plot
    scatter = ax.scatter(x_data,
                         y_data,
                         c=z,
                         cmap=custom_cmap,
                         s=90,  # Increase point size
                         alpha=0.9,
                         edgecolors='none')

    # Set axis ranges
    ax.set_xlim(0, xlim)
    ax.set_ylim(y_min, y_max)
    ax.set_yscale('log')

    # Optimize grid configuration
    ax.grid(True, which='major', linestyle='-', linewidth=1.2, alpha=0.3, color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth=0.8, alpha=0.2, color='gray')

    # Increase axis line width
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', width=1.5, length=10)
    ax.tick_params(axis='both', which='minor', width=1.0, length=6)

    # Title settings - keep only layer and fold numbers
    title = f"Layer {iLayer + 1}"
    ax.set_title(title, fontweight='medium', pad=30)  # Increase title padding

    # Set axis labels - display only under specified conditions
    # Only add x-axis information for Layer 5
    if iLayer == 4:  # 0-indexed, so Layer 5 is index 4
        ax.set_xlabel("Edge Parameter", labelpad=20)
    else:
        ax.set_xlabel("", labelpad=20)

    # Only add y-axis information for 'origin' type plots
    if fig_type == "origin":
        ax.set_ylabel("Normalized Distribution", labelpad=20)
    else:
        ax.set_ylabel("", labelpad=20)

    # Optimize colorbar
    cbar = plt.colorbar(scatter, pad=0.03)  # Increase colorbar padding
    cbar.set_label('', labelpad=20)  # Remove label
    cbar.ax.tick_params(labelsize=48)  # Update colorbar tick font size

    # Add threshold line, with increased line width
    ax.axvline(x=minKL_threshold,
               color='#1f77b4',
               linestyle='-',
               linewidth=6.0,  # Thicken the line width
               alpha=0.7,
               )

    # Threshold line label text - move to the left of the line
    text_y_pos = np.sqrt(1e-7 * ymax)  # Logarithmic center of y-axis
    ax.text(minKL_threshold - xlim * 0.01,  # Move to the left of the line
            text_y_pos,
            'Threshold',  # Simplified label
            fontsize=30,
            color='#1f77b4',
            verticalalignment='center',
            horizontalalignment='right',  # Right-align to bring text closer to the line
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=8))  # Increase text box padding

    # Adjust plot layout to provide more space for large fonts
    plt.tight_layout(pad=4.0)  # Increase overall padding

    # Adjust title position to avoid overflow
    title_obj = ax.title
    title_obj.set_position([.5, 1.04])

    # Save the plot - keep original file naming
    save_dir = os.path.join('curFeature_graph', f'fold_{ifold + 1}', f'layer_{iLayer + 1}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'distribution_{fig_type}.png')  # Keep original filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Restore default font settings to avoid affecting other plots
    plt.rcdefaults()
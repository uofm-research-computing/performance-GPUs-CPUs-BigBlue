import sys
import numpy as np
import matplotlib.pyplot as plt

def csv_to_heatmap(
    csv_path,
    output_path="heatmap.png",
    cmap="inferno"
):
    """
    Reads a CSV file and generates a heat map image.

    Assumes:
    - Values are normalized in [0, 1]
    - CSV represents a 2D grid
    """

    # Load CSV data
    data = np.loadtxt(csv_path, delimiter=",")

    # Create the heat map
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        data,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="auto"
    )

    # Add colorbar (scale)
    cbar = plt.colorbar(im)
    cbar.set_label("Normalized Temperature")

    # Axes labels
    plt.xlabel("X Index")
    plt.ylabel("Y Index")
    plt.title("Heat Map")

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Heat map saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_to_heatmap.py input.csv [output.png]")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "heatmap.png"

    csv_to_heatmap(csv_file, output_file)

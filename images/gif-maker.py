from pathlib import Path
from PIL import Image

image_dir = Path(".")

def create_dummy(N=50):
  """ A moving sine wave to have some pngs for debugging """

  import numpy as np
  from matplotlib import pyplot as plt

  phases = np.linspace(0, np.pi * 2, N)
  x_vals = np.linspace(0, 10, 100)

  for i in range(N):

    y_vals = np.sin(x_vals + phases[i])

    plt.figure()

    plt.plot(x_vals, y_vals)

    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.grid()

    plt.tight_layout()
    plt.savefig(image_dir / f"sinus_{i:04}.png", bbox_inches='tight', pad_inches=0.025)
    plt.close()

  return

def make_to_gif():
  """ Loads the pngs and creates the gif """

  files = sorted([file.name for file in image_dir.iterdir()])
  frames = list(map(lambda file: Image.open(image_dir / file), files))
  frame_one = frames[0]

  frame_one.save(image_dir / "awesome_haxball_agent_in_action.gif",
                 format="GIF",
                 append_images=frames,
                 loop=0,  # Loop forever
                 duration=1000 / 30,  # 30 FPS = 33.3 ms duration per frame
                 save_all=True)

  return

if __name__ == "__main__":
  make_to_gif()

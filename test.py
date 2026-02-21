import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Example data (replace with your real data)
x1 = np.array([1,2,3,4,5])
y1 = np.array([1,2,3,4,5])

x2 = np.array([1,2,3,4,5])
y2 = np.array([1,1.5,2.5,4.5,5])

x3 = np.array([1,2,3,4,5])
y3 = np.array([0.5,2.5,3.5,3.5,5])

datasets = [(x1,y1), (x2,y2), (x3,y3)]

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    x, y = datasets[frame]
    ax.scatter(x, y)
    ax.set_xlim(0,6)
    ax.set_ylim(0,6)
    ax.set_xlabel("True N")
    ax.set_ylabel("Human Count")
    ax.set_title(f"Dataset {frame+1}")

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(datasets),
    interval=1500,   # milliseconds per frame
    repeat=True
)

ani.save("scatter_animation.gif", writer="pillow", fps=1)

plt.close()
import matplotlib.pyplot as plt

# Class names
classes = [
    "car","bicycle","motorcycle","truck","other-vehicle","person",
    "bicyclist","motorcyclist","road","parking","sidewalk","other-ground",
    "building","fence","vegetation","trunk","terrain","pole","traffic-sign"
]

# Per-class mIoU
my_model = [94.6,51.2,48.0,42.7,46.8,64.4,53.7,28.3,92.0,68.1,76.6,33.2,92.7,68.8,85.3,69.5,71.4,59.3,66.9]
# my_model = [95.8,56.2,51.0,7.8,50.6,64.4,67.0,1.0,93.2,71.8,80.6,31.6,90.9,67.3,85.1,69.3,70.0,60.1,65.4]
sota =     [96.7,59.9,61.1,50.1,58.3,71.0,71.0,40.5,90.6,75.0,75.8,35.3,92.8,71.0,85.6,69.3,70.2,61.4,69.1]

# Delta: My Model − SOTA
delta = [m - s for m, s in zip(my_model, sota)]

# Plot
plt.figure(figsize=(18,6))
plt.bar(range(len(classes)), delta)
plt.axhline(0)

plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
plt.ylabel("Δ mIoU (My Model − SOTA)")
plt.title("Per-class mIoU Delta vs SOTA")

plt.tight_layout()
plt.show()

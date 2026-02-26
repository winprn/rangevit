import matplotlib.pyplot as plt
import numpy as np

# Data
classes = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
    "other-ground", "building", "fence", "vegetation", "trunk",
    "terrain", "pole", "traffic-sign"
]

percentages = [
    6.4721, 0.0520, 0.0725, 0.1067, 0.4661, 0.1000,
    0.0644, 0.0048, 18.4504, 1.2475, 12.6438,
    0.0960, 11.9304, 2.6518, 30.6103, 1.1543,
    13.4473, 0.3495, 0.0800
]

miou = [
    0.9325181760694745, 0.47129230410680695, 0.6270670992458783,
    0.7712425068035673, 0.4447274942110448, 0.6925891623787372,
    0.7859604741169315, 0.0, 0.9542786667092035, 0.5481967067855089,
    0.8267263959019384, 0.0312695660067791, 0.9037758885470917,
    0.639058470702075, 0.8766029867096834, 0.6744255134611402,
    0.7479312507723067, 0.6258694090731959, 0.47379097714237034
]

# Combine and sort data based on percentages (descending)
combined = sorted(zip(classes, percentages, miou), key=lambda x: x[1], reverse=True)
sorted_classes, sorted_percentages, sorted_miou = zip(*combined)

x = np.arange(len(sorted_classes))
width = 0.35

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Left axis: class distribution (Blue)
ax1.bar(x - width/2, sorted_percentages, width, label="Class Distribution (%)", color='steelblue')
ax1.set_ylabel("Class Distribution (%)", color='steelblue', fontweight='bold')
ax1.set_ylim(0, 35)
ax1.tick_params(axis='y', labelcolor='steelblue')

# Right axis: mIoU (Orange)
ax2 = ax1.twinx()
ax2.bar(x + width/2, sorted_miou, width, label="mIoU", color='darkorange')
ax2.set_ylabel("mIoU", color='darkorange', fontweight='bold')
ax2.set_ylim(0, 1.0)
ax2.tick_params(axis='y', labelcolor='darkorange')

# X-axis setup
ax1.set_xticks(x)
ax1.set_xticklabels(sorted_classes, rotation=45, ha='right')

# Title and layout
plt.title("Class Distribution vs mIoU (Sorted by Frequency)", fontsize=14)
fig.tight_layout()

plt.show()
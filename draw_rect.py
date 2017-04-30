import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

f = open('faces.txt', 'r')
l = f.readlines()
img = Image.open('twitter.png')
img = img.resize((128,128))
img.save('test.png')
img = np.array(Image.open('test.png'))
img = np.flipud(plt.imread('test.png'))
# Create figure and axes
fig,ax = plt.subplots(1)
#ax.gca().invert_yaxis()
# Display the image
ax.imshow(img, origin = 'lower')
size_box = 32
# Create a Rectangle patch
for line in l:
   coords = line.split(',')
   y_start, x_start = int(coords[0]), int(coords[1])
   rect = patches.Rectangle((x_start,y_start),size_box,size_box,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
   ax.add_patch(rect)

plt.show()

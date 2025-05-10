from PIL import Image
import matplotlib.pyplot as plt

# Specify the path
image_path = './abo-images-small/images/small/69/69465965.jpg'

# Open the image
img = Image.open(image_path)

# Display the image
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()

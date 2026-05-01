from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 8] 
#plt.rcParams['text.usetex'] = True

# Set parameters
RESO_MULTIPLIER = 10
WIDTH, HEIGHT = RESO_MULTIPLIER*1080, RESO_MULTIPLIER*720
MIN, MAX = -2, 2
DISTANCE = np.abs(MAX - MIN) 
THRESH_DISTANCE = 2
ITERATIONS = 250

def cropCenter(image, crop_width_percent, crop_height_percent):
    img_width, img_height = image.size
    
    # Calculate the coordinates for the crop box
    left = (img_width - crop_width_percent*img_width) // 2
    top = (img_height - crop_height_percent*img_height) // 2
    right = left + crop_width_percent*img_width
    bottom = top + crop_height_percent*img_height
    
    # Perform the crop
    return image.crop((left, top, right, bottom))

def pixelToGrid_x(i: int) -> float:
    return MIN + ((i * DISTANCE) / WIDTH)

def pixelToGrid_y(i: int) -> float:
    return MAX - ((i * DISTANCE) / HEIGHT)

def cexp(x: float) -> complex:
    return complex(np.cos(x), np.sin(x))

def functionalInput(z: complex, z_init: complex) -> float:
    return (z * z) + z_init
    # return z*(1+z)**2 + z_init

def generateImage(f: callable, z_init: complex, pix: object) -> None:
    for xi in range(0, WIDTH):
        real = pixelToGrid_x(xi)
        for yi in range(0, HEIGHT):
            imag = pixelToGrid_y(yi)
            z = f(complex(real, imag), z_init)
            count = 0
            while (abs(z) <= THRESH_DISTANCE) and (count < ITERATIONS + 1):
                z  = f(z, z_init)
                count += 1
            pix[xi, yi] = count
    return None 

z_init = complex(-1.3723793, -0.0110073) #complex(-1.3723793_029785152, -0.0110073_08959960493)
image = Image.new("L", (WIDTH, HEIGHT))
pix = image.load() 
generateImage(functionalInput, z_init, pix)
img_cropped = cropCenter(image, 0.9, 0.35)

fig, ax = plt.subplots()
ax.imshow(img_cropped, cmap = 'binary')

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor('white')

annotation_text = r"$c = {:.7f} {:+.7f}i$".format(z_init.real, z_init.imag)
ax.annotate(annotation_text, xy=(0.98, 0.1), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=12, color='black')


plt.savefig('julia_00.png', dpi = 300)
plt.show()
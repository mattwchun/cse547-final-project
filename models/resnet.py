from img_to_vec import Img2Vec # library to extract features using res-net
from PIL import Image

# Initialize Img2Vec with GPU
img2vec = Img2Vec()

# Read in an image
img = Image.open('v_i_frame_0.jpg')
# Get a vector from img2vec
vec = img2vec.get_vec(img)
print(vec)

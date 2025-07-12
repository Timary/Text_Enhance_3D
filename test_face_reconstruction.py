from PIL import Image
from face_reconstruction import reconstruct_3d_face

# Replace 'path_to_image.jpg' with your image file path
image = Image.open('TestSamples/AFLW2000/image00480.jpg')

# To display the image (optional)
# image.show()

print(reconstruct_3d_face(input_image=image))


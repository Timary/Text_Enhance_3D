from scripts.crop_align_face_with_return import align_face
from gradio_codeformer_utils import enhance_face_pil, inpaint_face_pil
from PIL import Image

img = Image.open('inputs/whole_imgs/00.jpg')
img.show()

img = align_face(img)
img.show()

img = enhance_face_pil(img)
img.show()

img = inpaint_face_pil(img)
img.show()
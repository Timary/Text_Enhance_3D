# test environment
python test_face_reconstruction.py
rm -r 'test'
mkdir 'test'
python scripts/crop_align_face.py -i 'inputs/whole_imgs' -o 'test'
python inference_codeformer.py -w 0.5 --has_aligned --input_path 'test'
python inference_inpainting.py --input_path 'inputs/masked_faces'
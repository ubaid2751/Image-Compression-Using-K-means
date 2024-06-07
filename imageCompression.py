import numpy as np 
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgba2rgb
from Kmeans import *
import sys
import os

def load_and_resize_image(image):
    try:
        # original_img = plt.imread(image)
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
        resized_img = resize(image, (256, 256), anti_aliasing=True)
        return resized_img
    except Exception as e:
        print(f"Error loading and resizing image: {e}")
        return None

def compress_image(image: np.array, n_colors, progress_bar):
    image = load_and_resize_image(image)
    if image.shape[-1] != 3:
        print(f"Error: Expected an RGB image with 3 channels but got shape {image.shape}")
        sys.exit(1) 
        
    X_img = np.reshape(image, (image.shape[0] * image.shape[1], 3))
    
    K = n_colors
    max_iters = 10

    model = K_means()
    initial_centroids = model.kmeans_init_centroids(X_img, K)
    centroids, idx = model.run_kmeans(X_img, K, progress_bar, initial_centroids, max_iters=max_iters)

    print(idx.shape)

    idx = model.find_closest_centroids(X_img, centroids)
    X_recovered = centroids[idx, :]
    X_recovered = np.reshape(X_recovered, image.shape)
    
    return X_recovered

def show_compressed_image(resized_img, X_recovered, K):
    fig, ax = plt.subplots(1,2, figsize=(16,16))
    plt.axis('off')

    ax[0].imshow(resized_img)
    ax[0].set_title('Original')
    ax[0].set_axis_off()

    ax[1].imshow(X_recovered)
    ax[1].set_title('Compressed with %d colours'%K)
    ax[1].set_axis_off()

    plt.show()
    
def save_compressed_image(image, output_path):
    plt.imsave(output_path, image)
    
# path = input("Enter image: ")
# image = load_and_resize_image(path)
# new_image = to_compress(image)
# show_compressed_image(image, new_image, 16)

# new_image = resize(new_image, image.shape, anti_aliasing=True)

# script_dir = os.path.dirname(os.path.abspath(__file__))
# output_path = os.path.join(script_dir, "compressed_image.jpg")
# save_compressed_image(new_image, output_path)
# print(f"Compressed image saved to {output_path}")
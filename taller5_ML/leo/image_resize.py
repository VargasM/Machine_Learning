import cv2
import sys
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: ", sys.argv[0], " original_size image_resize input_data")
        exit(-1)

    resized_images = []
    original_size = int(sys.argv[1])
    new_size = int(sys.argv[2])
    data = np.loadtxt(sys.argv[3], delimiter=',')

    for i in range(data.shape[0]):
        image = np.uint8(data[i, :-1].reshape(original_size, original_size))
        new_image = cv2.resize(image, (new_size, new_size), cv2.INTER_LANCZOS4)
        resized_images.append(np.append(new_image.flatten(), [data[i, -1]]))

    resized_images = np.array(resized_images)
    filename = './input_data/input_data_' + str(new_size) + '_' + str(new_size) + '.csv'
    np.savetxt(filename, resized_images, delimiter=',', newline='\n', fmt='%u')

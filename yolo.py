import os
import glob as glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import requests
import subprocess

TRAIN = True
REQUIREMENTS_SATISFIED = False
random_seed = 42
# Number of epochs to train for.
EPOCHS = 50


def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)
    else:
        print('File already present, skipping download...')
              
if(REQUIREMENTS_SATISFIED != True): 
    #clone yolov5 repo
    if not os.path.exists('yolov5'):
        subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'])

    os.chdir('yolov5')
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

#directory to save results
def set_res_dir():
    if not os.path.exists('runs/train'):
        os.makedirs('runs/train')
    res_dir_count = len(os.listdir('runs/train'))
    print(f"Current number of result directories: {res_dir_count}")
    if TRAIN:
        RES_DIR = f"results_{res_dir_count + 1}"
        print(RES_DIR)
    else:
        RES_DIR = f"results_{res_dir_count}"
    return RES_DIR

#result directory
RES_DIR = set_res_dir()

command = [
    'python', 'train.py',
    '--data', '../data.yaml',
    '--weights', 'yolov5s.pt',
    '--img', '640',
    '--epochs', str(EPOCHS),
    '--batch-size', '16',
    '--name', RES_DIR
]

subprocess.run(command)

class_names = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

#convert bounding boxes to yolov5 formatbbox
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels):
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        #denormalize coordinates
        xmin, ymin, xmax, ymax = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        width = xmax - xmin
        height = ymax - ymin
        
        class_name = class_names[labels[box_num]]
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = colors[class_names.index(class_name)], thickness=2)
        
        font_scale = min(1, max(3, int(w/500))) 
        font_thickness = min(2, max(10, int(w/50)))
        
        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        
        #text width and height
        tw, th = cv2.getTextSize(class_name, 0, fontScale=font_scale, thickness=font_thickness//2)[0]
        p2 = p1[0] + tw + 3, p1[1] - th - 10
        cv2.rectangle(image, p1, p2, color=colors[class_names.index(class_name)], thickness=-1)
        cv2.putText(image, class_name, (xmin+1, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=font_thickness)
    
    return image
    
#plot images with their bounding boxes
def plot_images(image_paths, label_paths, num_samples):
    all_train_images = glob.glob(image_paths)
    all_train_labels = glob.glob(label_paths)
    all_train_images.sort()
    all_train_labels.sort()
    
    num_images = len(all_train_images)
    
    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = np.random.randint(0, num_images)
        image = cv2.imread(all_train_images[j])
        with open(all_train_labels[j], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                xc, yc, w, h = bbox_string.split(' ')
                xc = float(xc)
                yc = float(yc)
                w = float(w)
                h = float(h)
                bboxes.append([xc, yc, w, h])
                labels.append(label)
        image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(image)
        plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    plt.show()
  
#visualize training images
#plot_images('../train/images/*', '../train/labels/*', 9)


#load tensorboard
def monitor_tensorboard():
    subprocess.run(['pip', 'install', 'tensorboard'])  # Install TensorBoard if needed
    subprocess.run(['pip', 'install', 'pyngrok'])  # Install pyngrok if needed
    subprocess.run(['tensorboard', '--logdir', 'runs/train'])  # Start TensorBoard

monitor_tensorboard()
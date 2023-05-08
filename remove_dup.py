import os
#remove every other picture and label
dirs = ['train', 'test', 'valid']
for i, dir in enumerate(dirs):
    all_image_names = sorted(os.listdir(dir+'/images/'))
    for j, image_name in enumerate(all_image_names):
        if j%2 == 0:
            file = image_name.split('.jpg')[0]
            os.remove(dir+'/images/'+image_name)
            os.remove(dir+'/labels/'+file+'.txt')

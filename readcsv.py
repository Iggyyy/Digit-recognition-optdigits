
class GetData:
    import csv, numpy as np

    data = []
    with open("set_dig.csv", newline='') as file:
        data = list(csv.reader(file))

    heads = data.pop(0)
    #print(data[0])
    imgs  = [[float(x) for x in i] for i in data]
    labels = np.zeros((len(imgs), 10))

    

    for i,k in enumerate(imgs):
        labels[i][ int(imgs[i][64]) ] = 1
        del imgs[i][64]


    imgs = np.array(imgs)

   

   
    #Normalization
    maxes = np.max(imgs, axis=0)
    idx = np.argwhere(maxes==0)

    imgs = np.delete(imgs, idx, axis=1)
    maxes = np.delete(maxes, idx)
    
    for i in range(len(imgs[0])):
        imgs[:,i] /= maxes[i]

    
   
    
    cut = int(len(imgs)*0.25)

    t_imgs = imgs[0:1000]
    t_labels = labels[0:1000]

    imgs = imgs[1000:2000]
    labels = labels[1000:2000]

   
    

   
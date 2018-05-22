import numpy as np

class VocLabelsProcessor():
    def __init__(self, num_classes):
        
        self.contour_color =  np.array([224, 224, 192])
        self.contour_class = num_classes - 1
        
        self.num_classes = num_classes
        self.cmap = self.color_map(num_classes - 1)

    def color_map(self, num_classes, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)
    
        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((num_classes, 3), dtype=dtype)
        for i in range(num_classes):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3
    
            cmap[i] = np.array([r, g, b])
    
        cmap = cmap/255 if normalized else cmap
        return cmap

    def color_label_to_class(self, label):
        
        result = np.zeros((label.shape[0], label.shape[1]), dtype = np.uint8)
        
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                ok = False
                
                for index in range(len(self.cmap)):
                    if np.array_equal(label[i,j],self.cmap[index]):
                        result[i,j] = index
                        ok = True
                        
                if not ok:
                    result[i,j] = self.contour_class
                
        return result
    
    def class_label_to_color(self, label):
        
        result = np.zeros((label.shape[0], label.shape[1], 3), dtype = np.uint8)
        
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                
                if label[i,j] < len(self.cmap):
                    result[i,j] = self.cmap[label[i,j]]
                else:
                    result[i,j] = self.contour_color
                    
        return result
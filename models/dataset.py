import torch
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import graph as g
import math

class CreateImageDataset(Dataset):
    def __init__(self, labels, img_dir, image_size, image_column, output_columns, input_columns=None, add_graphs=None, concat_graphs=None, transform=None, target_transform=None, train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = labels
        self.image_column = image_column
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.train = train
        self.add_graphs = add_graphs
        self.concat_graphs = concat_graphs
        self.image_size = image_size

        self.first = True

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        output_label = torch.tensor(self.img_labels.iloc[idx, self.output_columns], dtype=torch.float32)
        if self.input_columns is not None:
            input_label = torch.tensor(self.img_labels.iloc[idx, self.input_columns], dtype=torch.float32)
        else:
            input_label = None
        grade = self.img_labels.iloc[idx]['Rank']
        name = self.img_labels.iloc[idx, self.image_column]
        img_path = os.path.join(self.img_dir, name)
        image = Image.open(img_path)
        
        #------------------------------------------------------------------
        tensor_trans = transforms.Compose([
            transforms.Resize([self.image_size,self.image_size]),
            transforms.ToTensor(),
        ])
        resize = transforms.Resize([self.image_size,self.image_size])
        #------------------------------------------------------------------
        if self.transform:
            if self.train:
                image = self.transform(image)
            else:
                image = tensor_trans(image)
            
            image = self.add_graph(image, tensor_trans, grade, img_path)
            image = self.concat_graph(image,tensor_trans, grade, img_path)
            resize(image)

        if self.target_transform:
            label = self.target_transform(label)
            
        if input_label is not None:
            input = {'image':image, 'input_label':input_label}
        else:
            input = {'image':image}
        #------------------------------------------------------------------         

        return input, output_label, name
    

    def add_graph(self, image, transform, grade, img_path):
        add_graphs = []
        
        if self.add_graphs is None:
            return image
        
        for graph in self.add_graphs:
            if graph == 'color':
                temp = g.colorGraph(img_path)
            elif graph == 'gray':
                temp = g.grayGraph(img_path)
            elif graph == 'gcolor':
                temp = Image.open(f'kde/kde_Color_{grade}.png')
            elif graph == 'gsurface':
                temp = Image.open(f'kde/kde_Surface Moisture_{grade}.png')
            elif graph == 'gtexture':
                temp = Image.open(f'kde/kde_Texture_{grade}.png')
            elif graph == 'gmarbling':
                temp = Image.open(f'kde/kde_Marbling_{grade}.png')
            elif graph == 'gtotal':
                temp = Image.open(f'kde/kde_Total_{grade}.png')
            else:
                print('invalid graph name')

            temp = transform(temp)
            add_graphs.append(temp)

        for graph in add_graphs:
            image += graph[0:3]

        return image
    
    def concat_graph(self, image, transform, grade, img_path):
        concat_graphs = []
        toImage = transforms.ToPILImage()
        if self.add_graphs is None:
            return image
        
        for graph in self.concat_graphs:
            if graph == 'color':
                temp = g.colorGraph(img_path)
            elif graph == 'gray':
                temp = g.grayGraph(img_path)
            elif graph == 'gcolor':
                temp = Image.open(f'kde/kde_Color_{grade}.png')
            elif graph == 'gsurface':
                temp = Image.open(f'kde/kde_Surface Moisture_{grade}.png')
            elif graph == 'gtexture':
                temp = Image.open(f'kde/kde_Texture_{grade}.png')
            elif graph == 'gmarbling':
                temp = Image.open(f'kde/kde_Marbling_{grade}.png')
            elif graph == 'gtotal':
                temp = Image.open(f'kde/kde_Total_{grade}.png')
            else:
                print('invalid graph name')
            
            temp = transform(temp)
            temp = toImage(temp[0:3])
            concat_graphs.append(temp)
            
        image = toImage(image)
        w = math.ceil(math.sqrt(len(concat_graphs)))  
        result_image = Image.new('RGB', (self.image_size * w, self.image_size * w))
        result_image.paste(image,(0,0,self.image_size,self.image_size))
        for i, graph in enumerate(concat_graphs):
            row = (i+1) // w
            col = (i+1) % w
            result_image.paste(graph,(row*self.image_size, col*self.image_size, (row+1)*self.image_size, (col+1)*self.image_size))
        result_image = transform(result_image)
        return result_image


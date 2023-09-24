import torch
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class CreateImageDataset(Dataset):
    def __init__(self, labels, img_dir, dataset_cfgs, output_columns, train=True):

        self.train_transforms = []
        self.test_transforms = []
        self.image_sizes = []
        self.isImage = []
        self.input_columns = []
        self.graph = []
        self.output_columns = output_columns
        self.model_cnt = len(dataset_cfgs)

        for dataset_cfg in dataset_cfgs:
            image_size = dataset_cfg['image_size']
            isImage = dataset_cfg['isImage']
            input_column = dataset_cfg['input_column']
            graph = dataset_cfg['graph']
            
            
            train_transform = transforms.Compose([
            transforms.Resize([image_size,image_size]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((-180,180)),
            transforms.ToTensor(),
            ])

            test_transform = transforms.Compose([
            transforms.Resize([image_size,image_size]),
            transforms.ToTensor(),
            ])

            self.train_transforms.append(train_transform)
            self.test_transforms.append(test_transform)
            self.image_sizes.append(image_size)
            self.isImage.append(isImage)
            self.graph.append(graph)
            self.input_columns.append(input_column)

        self.img_dir = img_dir
        self.img_labels = labels
        self.train = train

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        inputs = []
        outputs = torch.tensor(self.img_labels.iloc[idx, self.output_columns], dtype=torch.float32)
        for i in range(self.model_cnt):
            if self.isImage[i] == True:
                name = self.img_labels.iloc[idx, self.input_columns[i]]
                img_path = os.path.join(self.img_dir, name)
                image = Image.open(img_path)

                if self.train:
                    image = self.train_transforms[i](image)
                else:
                    image = self.test_transforms[i](image)
                inputs.append(image)
                
            elif self.graph is not None:
                grade = self.img_labels.iloc[idx]['Rank']
                graph = self.openGraph(self.graph[i], grade)
                graph = self.test_transforms[i](graph)
                inputs.append(graph[0:3])

            else:
                input = torch.tensor(self.img_labels.iloc[idx, self.input_columns[i]], dtype=torch.float32)
                inputs.append(input) 

        return inputs, outputs, name
    
    
    
    def openGraph(self, graph, grade):
        if graph == 'gcolor':
            temp = Image.open(f'utils/kde/kde_Color_{grade}.png')
        elif graph == 'gsurface':
            temp = Image.open(f'utils/kde/kde_Surface Moisture_{grade}.png')
        elif graph == 'gtexture':
            temp = Image.open(f'utils/kde/kde_Texture_{grade}.png')
        elif graph == 'gmarbling':
            temp = Image.open(f'utils/kde/kde_Marbling_{grade}.png')
        elif graph == 'gtotal':
            temp = Image.open(f'utils/kde/kde_Total_{grade}.png')
        else:
            print('invalid graph name')
        return temp

    
    # def add_graph(self, image, transform, grade, img_path):
    #     add_graphs = []
        
    #     if self.add_graphs is None:
    #         return image
        
    #     for graph in self.add_graphs:
    #         if graph == 'color':
    #             temp = g.colorGraph(img_path)
    #         elif graph == 'gray':
    #             temp = g.grayGraph(img_path)
    #         elif graph == 'gcolor':
    #             temp = Image.open(f'kde/kde_Color_{grade}.png')
    #         elif graph == 'gsurface':
    #             temp = Image.open(f'kde/kde_Surface Moisture_{grade}.png')
    #         elif graph == 'gtexture':
    #             temp = Image.open(f'kde/kde_Texture_{grade}.png')
    #         elif graph == 'gmarbling':
    #             temp = Image.open(f'kde/kde_Marbling_{grade}.png')
    #         elif graph == 'gtotal':
    #             temp = Image.open(f'kde/kde_Total_{grade}.png')
    #         else:
    #             print('invalid graph name')

    #         temp = transform(temp)
    #         add_graphs.append(temp)

    #     for graph in add_graphs:
    #         image += graph[0:3]

    #     return image
    
    # def concat_graph(self, image, transform, grade, img_path):
    #     concat_graphs = []
    #     toImage = transforms.ToPILImage()
    #     if self.add_graphs is None:
    #         return image
        
    #     for graph in self.concat_graphs:
    #         if graph == 'color':
    #             temp = g.colorGraph(img_path)
    #         elif graph == 'gray':
    #             temp = g.grayGraph(img_path)
    #         elif graph == 'gcolor':
    #             temp = Image.open(f'kde/kde_Color_{grade}.png')
    #         elif graph == 'gsurface':
    #             temp = Image.open(f'kde/kde_Surface Moisture_{grade}.png')
    #         elif graph == 'gtexture':
    #             temp = Image.open(f'kde/kde_Texture_{grade}.png')
    #         elif graph == 'gmarbling':
    #             temp = Image.open(f'kde/kde_Marbling_{grade}.png')
    #         elif graph == 'gtotal':
    #             temp = Image.open(f'kde/kde_Total_{grade}.png')
    #         else:
    #             print('invalid graph name')
            
    #         temp = transform(temp)
    #         temp = toImage(temp[0:3])
    #         concat_graphs.append(temp)
            
    #     image = toImage(image)
    #     w = math.ceil(math.sqrt(len(concat_graphs)))  
    #     result_image = Image.new('RGB', (self.image_size * w, self.image_size * w))
    #     result_image.paste(image,(0,0,self.image_size,self.image_size))
    #     for i, graph in enumerate(concat_graphs):
    #         row = (i+1) // w
    #         col = (i+1) % w
    #         result_image.paste(graph,(row*self.image_size, col*self.image_size, (row+1)*self.image_size, (col+1)*self.image_size))
    #     result_image = transform(result_image)
    #     return result_image


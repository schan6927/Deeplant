import mlflow
from torchvision import transforms

def log_input(dataset):

    input, output = dataset[0]

    if 'input_label' in input:
        input_label = input['input_label']
    else:
        input_label = None

    image = input['image']
    trans_image = transforms.ToPILImage()
    _image = trans_image(image)
    mlflow.log_image(_image,'input/input_image.jpg')
    mlflow.log_input(dataset=dataset)

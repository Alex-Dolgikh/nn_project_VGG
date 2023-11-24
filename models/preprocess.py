from torchvision import transforms as T

process = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
    ]
)

def preprocess(img):
    return process(img)

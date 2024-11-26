import torch
from torch import nn
from tqdm import tqdm
from torch import optim

class ResNet(nn.Module):
    def __init__(self, name : str = 'resnet50', pt : bool = True, transform = None, lr = 0.001):
        super().__init__()

        if name != 'resnet18' and name != 'resnet34' and name != 'resnet50' and name != 'resnet101' and name != 'resnet152':
            print(f'Error name needs to be: resnet18, resnet34, resnet50, resnet101 or resnet152"')
            return
            
        self.model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained = pt)

        self.pt = pt
        self.name = name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.transform = transform       

    def __call__(self, image, label):
        self.model.eval()
        self.model.to(self.device)
    
        if isinstance(image, torch.Tensor):
            img = image
        else:
            if self.transform:
                img = self.transform(image)
    
        img = img.unsqueeze(0)
        img = img.to(self.device)
    
        with torch.no_grad():
            output = self.model(img)
    
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
        return probabilities[label].item()

    def fit(self, train_dl, valid_dl, path = './model.pt', epochs=10):
        history = dict()
        history['train_loss'] = list()
        history['train_accuracy'] = list()
        history['eval_loss'] = list()
        history['eval_accuracy'] = list()
        
        self.model.to(self.device)

        best_eval = -1.0
        
        for epoch in range(epochs):
            self.model.train()
            
            total = 0
            running_loss = 0.0
            running_accuracy = 0.0
            
            for data in tqdm(train_dl):
                img, label = data
                
                img, label = img.to(self.device), label.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(img)
                
                loss = self.criterion(outputs, label)
                
                loss.backward()
                
                self.optimizer.step()
                
                _, predicted = torch.max(outputs, 1)
                
                total += len(label)
                
                running_accuracy += (predicted == label).sum().item()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / total
            
            epoch_accuracy = (running_accuracy / total)
            
            valid_loss, valid_accuracy = self.validation(valid_dl)
            
            history['train_loss'].append(epoch_loss)
            history['train_accuracy'].append(epoch_accuracy)
            history['eval_loss'].append(valid_loss)
            history['eval_accuracy'].append(valid_accuracy)

            best_eval = self.save_model(best_eval, valid_accuracy, path)
            
            print(f"Epoch {epoch+1}/{epochs}, train_loss: {epoch_loss:.4f}, train_acc: {epoch_accuracy:.2f}, val_loss: {valid_loss:.4f}, val_acc: {valid_accuracy:.2f}")
        
        return history
        
    def validation(self, dataloader):        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for data in dataloader:
                img, label = data
                
                img, label = img.to(self.device), label.to(self.device)
                        
                outputs = self.model(img)
                
                loss = self.criterion(outputs, label)
                total_loss += loss.item()
                
                total += len(label)
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == label).sum().item()
    
        avg_loss = total_loss / total
        accuracy = (correct / total)
        
        return avg_loss, accuracy

    def test(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
    
        with torch.no_grad():
            for data in dataloader:
                img, label = data
                img, label = img.to(self.device), label.to(self.device)
                
                outputs = self.model(img)
                
                total += len(label)
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == label).sum().item()
        
        accuracy = 100 * (correct / total)
        
        return accuracy

    def save_model(self, best_eval, eval, path):
        if best_eval < eval:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, path)
            best_eval = eval
            
        return best_eval

    def load_model(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.model.to(device)
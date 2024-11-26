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
        """
        Performs a forward pass through the model to compute the probability of a given label for an input image.
        
        This method allows an instance of the class to behave like a function. It takes an image and a label, processes 
        the image (if necessary), performs a forward pass through the model, and returns the probability of the specified label.

        The model is set to evaluation mode and transferred to the appropriate device (CPU or GPU) before making predictions.
        If the input image is not already a tensor, it is transformed using the defined `transform` function before being passed 
        through the model.

        Parameters:
        - image: The input image to be evaluated. It can be a `torch.Tensor` or a format that can be transformed 
                (e.g., a PIL image or NumPy array).
        - label: The label (class index) for which the probability is to be computed. This should be an integer representing 
                the target class index.

        Returns:
        - The probability of the input image belonging to the given label as a float. The probability is obtained using 
        the softmax function, which normalizes the output of the model to a probability distribution over the classes.
        
        Notes:
        - The method uses `torch.no_grad()` to disable gradient calculation, as we are in evaluation mode and not performing 
        backpropagation.
        - The `transform` function, if defined, is used to preprocess the input image (e.g., resizing, normalization).
        """
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
        """
        Trains the model using the provided training and validation data loaders.

        This method performs the training over multiple epochs, where for each epoch, the model is trained using 
        the training dataset (`train_dl`) and evaluated using the validation dataset (`valid_dl`). It calculates 
        the loss and accuracy for both the training and validation datasets, and updates the model's weights 
        using backpropagation and optimization. 

        The best model (based on validation accuracy) is saved to the specified path. After each epoch, 
        training and validation statistics (loss and accuracy) are recorded in a history dictionary.

        Parameters:
        - train_dl: The data loader for the training dataset, providing batches of input data and labels.
        - valid_dl: The data loader for the validation dataset, used for evaluating the model after each epoch.
        - path: The file path where the best model (based on validation accuracy) will be saved. Default is './model.pt'.
        - epochs: The number of epochs to train the model. Default is 10.

        Returns:
        - history: A dictionary containing the training and validation loss and accuracy for each epoch:
            - 'train_loss': List of training loss values for each epoch.
            - 'train_accuracy': List of training accuracy values for each epoch.
            - 'eval_loss': List of validation loss values for each epoch.
            - 'eval_accuracy': List of validation accuracy values for each epoch.
        """
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
        """
        Evaluates the model on the provided data loader and returns the loss and accuracy.

        This method puts the model in evaluation mode, processes the data from the provided data loader (`dataloader`), 
        calculates the loss and accuracy for the dataset, and returns the average loss and accuracy. During evaluation, 
        gradient computation is disabled using `torch.no_grad()` to save memory and computation time. 

        Parameters:
        - dataloader: The data loader containing the dataset to evaluate the model on. It provides batches of input data 
        and corresponding labels.

        Returns:
        - avg_loss: The average loss for the dataset.
        - accuracy: The accuracy of the model on the dataset, defined as the proportion of correct predictions.

        Notes:
        - The model is set to evaluation mode using `self.model.eval()` to disable features like dropout and batch normalization.
        - The loss is accumulated using `total_loss`, and the accuracy is computed by comparing the predicted labels with 
        the actual labels.
        - The `self.criterion` is assumed to be the loss function used for evaluation (e.g., CrossEntropyLoss).
        - The function uses `torch.no_grad()` to ensure that gradients are not computed during the forward pass.
        """
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
        """
        Tests the model on the provided data loader and returns the accuracy.

        This method evaluates the model on the given dataset by computing the number of correct predictions 
        compared to the total number of samples. It disables gradient computation with `torch.no_grad()` to 
        save memory and computation time during the evaluation.

        Parameters:
        - dataloader: The data loader containing the dataset to evaluate the model on. It provides batches of input data 
        and corresponding labels.

        Returns:
        - accuracy: The accuracy of the model on the dataset, expressed as a percentage.

        Notes:
        - The model is set to evaluation mode using `self.model.eval()` to disable features like dropout and batch normalization.
        - The accuracy is computed by comparing the predicted labels with the actual labels and calculating the proportion of correct predictions.
        - The function assumes that the model uses `torch.max()` to determine the predicted class, and that `self.device` is properly set for computation (e.g., on a GPU or CPU).
        - The result is returned as a percentage, i.e., accuracy * 100.
        """
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
        """
        Saves the model and optimizer state if the current evaluation score is better than the previous best.

        This method checks if the current evaluation score (`eval`) is better than the previous best evaluation score 
        (`best_eval`). If so, it saves the model's and optimizer's state dictionaries to the specified file path (`path`), 
        and updates the `best_eval` to the new evaluation score.

        Parameters:
        - eval: The current evaluation score (e.g., validation accuracy or loss).
        - path: The file path where the model and optimizer state dictionaries will be saved.

        Returns:
        - best_eval: The updated best evaluation score.
        """
        if best_eval < eval:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, path)
            best_eval = eval
            
        return best_eval

    def load_model(self, path):
        """
        Loads the model and optimizer state from a saved checkpoint.

        This method loads the model and optimizer state dictionaries from a specified file path (`path`). 
        It restores the model weights and optimizer state, ensuring that the model is ready for further 
        training or evaluation. The model is then transferred to the appropriate device (GPU if available, 
        otherwise CPU).

        Parameters:
        - path: The file path to the saved checkpoint containing the model and optimizer state dictionaries.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.model.to(device)
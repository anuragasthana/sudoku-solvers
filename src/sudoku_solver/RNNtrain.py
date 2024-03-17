from util import *
from constants import *
from train import *
from dataLoader import *
import torch.optim as optim
import torch.nn as nn
import sys
import os
import numpy as np


def train(data: SudokuDataloaders, params: Hyperparams, device="cpu"):

    print("Training with hyperparameters:")
    print(params)
    
    # Create model
    model = SudokuRNN()
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(params.patience)

    # Extracting configuration parameters
    #LR = config["learning_rate"]
    #SAVE_EVERY = config["save_epoch"]
    #MODEL_TYPE = config["model_type"]
    #HIDDEN_SIZE = config["hidden_size"]
    #DROPOUT_P = config["dropout"]
    #SEQ_SIZE = config["sequence_size"]
    #CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(MODEL_TYPE, N_EPOCHS, HIDDEN_SIZE, DROPOUT_P)
    
    model = model.to(device) # Move model to the specified device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #Use Adam optimizer
    loss = nn.CrossEntropyLoss() # Use cross entropy loss for multi-class categorical problem

    # Lists to store training and validation losses over the epochs
    train_losses, validation_losses = [], []
    
    # Training over epochs
    for epoch in range(params.epochs):

        print(f"Epoch {epoch+1}/{params.epochs}")

        # TRAIN: Train model over training data
        for i in range(len(data)):
            
            model.init_hidden() # Zero out the hidden layer
            model.hidden_state = model.hidden_state.to(device)
            if MODEL_TYPE == "lstm":
                model.cell_state = model.cell_state.to(device)
            model.zero_grad()   # Zero out the gradient
            optimizer.zero_grad()
            random_sequence_losses = []
            random_character_losses = []
            random_song_sequence = get_random_song_slice(data[i], SEQ_SIZE)
            n = len(random_song_sequence)
            num_chars = len(char_idx_map)
            
            # Iterate over sequence characters
            for i in range(1, n):
                one_hot_ground_truth = char_to_deviced_one_hot(random_song_sequence[i], char_idx_map, device)
                input = characters_to_tensor(random_song_sequence[i-1], char_idx_map)
                input = input.to(device)
                
                #TODO: use activation for future heatmap plots
                output, activations = model(input)
                resultant_loss = loss(output, one_hot_ground_truth)
                random_character_losses.append(resultant_loss)
            #for char_loss in random_character_losses:
                #char_loss.backward()
                #model(char_idx_map[char])
                #One Hot Encode subsequence
            #random_character_losses /= (n-1)
            #Backpropagate accumulated loss
            total_loss_per_sequence = sum(random_character_losses)
            total_loss_per_sequence.backward()
            #Update weights                
            optimizer.step()
            #Average loss for the sequence
            avg_loss_per_sequence = total_loss_per_sequence/len(random_character_losses)
            random_sequence_losses.append(avg_loss_per_sequence)


            # Display progress
            msg = '\rTraining Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/len(data)*100, i, avg_loss_per_sequence)
            sys.stdout.write(msg)
            sys.stdout.flush()

        print()

        # Append the avg loss on the training dataset to train_losses list
        avg_dataset_loss = sum(random_sequence_losses)/len(random_sequence_losses)
        print(avg_dataset_loss.detach())
        train_losses.append(avg_dataset_loss.detach())

        
        # VAL: Evaluate Model on Validation dataset
        model.eval() # Put in eval mode (disables batchnorm/dropout) !
        
        #print("Start of Validation")

        #validation_losses_this_epoch = []
        with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
            # Iterate over validation data
            for i in range(len(data_val)):
                '''
                 
                    - For each song:
                        - Zero out/Re-initialise the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.) (Done for you)
                        - Get a random sequence of length: SEQ_SIZE from each song- Get a random sequence of length: SEQ_SIZE from each song (check util.py)
                        - Iterate over sequence characters : 
                            - Transfer the input and the corresponding ground truth to the same device as the model's
                            - Do a forward pass through the model
                            - Calculate loss per character of sequence
                        - Calculate avg loss for the sequence
                    - Calculate avg loss for the validation dataset and 
                '''

                model.init_hidden() # Zero out the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.)
                model.hidden_state = model.hidden_state.to(device)
                if MODEL_TYPE == "lstm":
                    model.cell_state = model.cell_state.to(device)
                #Finish next steps here
                random_sequence_losses = []
                random_character_losses = []
                
                random_song_sequence = get_random_song_slice(data_val[i], SEQ_SIZE)
                n = len(random_song_sequence)
                #num_chars = len(char_idx_map)
                
                for i in range(1,n):
                    one_hot_ground_truth = char_to_deviced_one_hot(random_song_sequence[i], char_idx_map, device)
                    input = characters_to_tensor(random_song_sequence[i-1], char_idx_map)
                    input = input.to(device)
                    # Run the model on the input sequence
                    output, activations = model(input)
                    
                    # Calculate the loss
                    resultant_loss = loss(output, one_hot_ground_truth)
                    random_character_losses.append(resultant_loss)

                avg_loss_per_sequence = sum(random_character_losses)/len(random_character_losses)
                random_sequence_losses.append(avg_loss_per_sequence)
                #validation_losses_this_epoch.append(avg_loss_per_sequence)

                # Display progress
                msg = '\rValidation Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/len(data_val)*100, i, avg_loss_per_sequence)
                sys.stdout.write(msg)
                sys.stdout.flush()

            print()


        # Append the avg loss on the validation dataset to validation_losses list
        avg_validation_loss = sum(random_sequence_losses)/len(random_sequence_losses)
        validation_losses.append(avg_validation_loss.detach())

        model.train() #TURNING THE TRAIN MODE BACK ON !

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        # Save checkpoint.
        if (epoch % SAVE_EVERY == 0 and epoch != 0)  or epoch == N_EPOCHS - 1:
            print('=======>Saving..')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, './checkpoint/' + CHECKPOINT + '.t%s' % epoch)

    
    return train_losses, validation_losses

# Batch convert to one-hot encoding
def char_to_deviced_one_hot(char, char_idx_map, device):
    num_chars = len(char_idx_map)
    
    char = char_idx_map[char]
    char = one_hot_encoding([char], num_chars)
    
    # Convert to tensor (2D)
    char = torch.tensor(char, dtype=torch.float32)
    char = char.to(device) #Transfer ground truth to same device as model
    return char

def one_hot_encoding(value):
    """
    
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have

    returns:
        oneHot : N X num_classes 2D array
    """
    result = [0] * 9
    if ((value == 0) or (value == None)):
        return result
    result[value-1]=1
    return result

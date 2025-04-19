# DA6401_Assignment_2

Wandb Report Link - https://wandb.ai/alokgaurav04-indian-institute-of-technology-madras/DA6401_Assignment_2/reports/DA6401-Assignment-2--VmlldzoxMjE5NzI5MA

Github Report Link - https://github.com/alokgaurav04/DA6401_Assignment_2

# PART - A : Training from scratch

   # Assignment_2_part_A.ipynb
   
    Code for CNN is provided in github with the name "Assignment_2_Part_A.ipynb"

    # I used google colab for training using GPU and uploaded the iNaturalist.zip file to google drive , hence I need to mount the drive first 
    from google.colab import drive
    drive.mount('/content/drive')   

    # Unzip the iNaturalist.zip file on google drive
    !unzip -q /content/drive/MyDrive/nature_12K.zip -d /content

    #iNaturalist_12k.zip file contains 2 folders ,train and val
    # So , I have divided train folder into train and val dataset in the ratio of 8:2 .
    # and I have used val folder as test dataset.

    # Set the directory path , if using local computer then this path needs to be changed
    DATA_DIR = "/content/inaturalist_12K"
    
    For Flexibility , the code for CNN has been coded is a way that the number of filter , size of filters and activation function can be changed .
    
    #Defining the CNN for classification
    model = CNNModel(num_classes=NUM_CLASSES,
                 filters=config.filters,
                 filter_policy=config.filter_policy,
                 activation=config.activation,
                 batch_norm=config.batch_norm,
                 dropout=config.dropout , kernel_size=3,dense_neurons=128).to(device)

    #Training the network :
    def sweep_train() function is provided to do training and validation 

    #Test set 
    Best Model is saved in wandb to perform classification on test set data.

    
# PART - B : Fine-tuning a pre-trained model

   # Assignment_2_Part_B.ipynb

      Code for fine-tuning is provided in github with the name "Assignment_2_Part_B.ipynb"      

      - Training and Validation is performed on ResNet50
      - All layers except last k layers are frozen (In my code k=2)
      - Best validation accuracy model is saved and test accuracy is evaluated on the best model
      - Test Accuracy is the final output 

      
       

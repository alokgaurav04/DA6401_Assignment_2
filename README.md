# DA6401_Assignment_2
Repository for Assignment - 2 of course DA6401

Wandb Report Link - https://wandb.ai/alokgaurav04-indian-institute-of-technology-madras/DA6401_Assignment_2/reports/DA6401-Assignment-2--VmlldzoxMjE5NzI5MA

Github Report Link - https://github.com/alokgaurav04/DA6401_Assignment_2

#PART - A : Training from scratch

   #Question - 1 :
   
    Code for CNN is provided in github with the name "Assignment_Part_A.ipynb"
    
    For Flexibility , the code for CNN has been coded is a way that the number of filter , size of filters and activation function can be changed .

        model = CNNModel(num_classes=NUM_CLASSES,
                 filters=config.filters,
                 filter_policy=config.filter_policy,
                 activation=config.activation,
                 batch_norm=config.batch_norm,
                 dropout=config.dropout).to(device)
       

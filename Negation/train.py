from Negation.mymodel import *
train_data = data_process('./bioscope_abstract.csv')
val_data = data_process('./bioscope_full.csv')
scope_model = ScopeModel(train=True)
scope_model.train(train_data, val_data)

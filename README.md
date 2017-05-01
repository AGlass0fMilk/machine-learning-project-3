The params.pickle file contains the weight matrix.  
To obtain the weight object run the line 

W = pickle.load( open( "params.pickle", "rb" ) )

and W will now contain the weight matrix.
import warnings
warnings.filterwarnings('ignore')
from category_encoders import LeaveOneOutEncoder,WOEEncoder
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostClassifier, Pool,
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import tensorflow as tensorflow
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
 
 
 
 
 
 # Create a class for all the models
class Model_class(object):
    def __init__(self, df:dict):
        self.df = df
        
        
    def split_data(self, historic:bool)->tuple:
        """
        Split the data into train and test sets, depending on whether the encdoder needs to be fited on the full data or not.

        Parameters
        ----------
        historic : bool
            False if the encoder needs to be fited on the Months April to February and to transform March
            True if the encoder needs to be fited on the Months April to December and to transform January to March

        Returns
        -------
        tuple
            df_train : dict
                Dataframe with training data.
            df_test : dict
                Dataframe with testing data.
        """
        # Get list of the months to train, remove months on which to test        
        if historic:
            months_to_train = list(range(1,13))
            # remove months 12,1,2,3 from the train set
            months_to_train = [i for i in months_to_train if i not in [12,1,2,3]]
        else:
            months_to_train = list(range(1,13))
            months_to_train.remove(3)
        k = max(self.df["order_item_id"])-1
        # Split into train and test. "~" in front of a variable means "not"
        df_train = self.df.loc[:k][self.df.loc[:k,"order_month"].isin(months_to_train)]
        df_test = self.df.loc[:k][~self.df.loc[:k,"order_month"].isin(months_to_train)]
        # Get the validation set
        # df_valid = self.df.iloc[k+1:, :]
        # Drop unnecessary columns
        columns_to_drop = ["order_date", "delivery_date", "user_dob", "user_reg_date", "order_id","order_item_id"]
        real_class = pd.read_csv("orders_realclass.txt", delimiter=";")
        df_valid = self.df.iloc[k+1:, :]
        df_valid.reset_index(drop=True,inplace=True)
        df_valid["return"] = real_class["returnShipment"]
        
        df_train.drop(columns_to_drop, axis=1, inplace=True)
        df_test.drop(columns_to_drop, axis=1, inplace=True)
        df_valid.drop(columns_to_drop, axis=1, inplace=True)
        return df_train, df_test, df_valid
    
    def LOE_Encoder(self, df_train:dict, df_test:dict, columns:list ,sig:float)->dict, dict, object:
        """
        Leave One Out Encoder to calculate the response variable for each category.

        Parameters
        ----------
        df_train : dict
            Dataframe with training data.
        df_test : dict
            Dataframe with testing data.
        columns : list
            Categorical columns to encode.
        sig : float
            Random noise added to the response variable.

        Returns
        -------
        df_encode_train : dict
            Dataframe with training data with encoded columns.
        df_encode_test : dict
            Dataframe with testing data with encoded columns.
        encoder : object
            Encoder object.
        """                      
        encoder = LeaveOneOutEncoder(cols=columns, return_df=True,sigma=sig)
        df_encode_train = encoder.fit_transform(df_train.drop(["return"],axis=1),df_train[["return"]]).round(3)
        df_encode_test = encoder.transform(df_test.drop(["return"],axis=1)).round(3)
        df_encode_train , df_encode_test = df_encode_train.join(df_train[["return"]]), df_encode_test.join(df_test[["return"]])
        return df_encode_train, df_encode_test, encoder
    
    def WOE_Encoder(self,df_train,df_test,columns,sig):
        encoder = WOEEncoder(cols=columns, return_df=True,sigma=sig, verbose=True,regularization=1)
        df_encode_train = encoder.fit_transform(df_train.drop(["return"],axis=1),df_train[["return"]])
        df_encode_test = encoder.transform(df_test.drop(["return"],axis=1))
        df_encode_train , df_encode_test = df_encode_train.join(df_train[["return"]]), df_encode_test.join(df_test[["return"]])
        return df_encode_train, df_encode_test, encoder
    
    def neural_network(self, df_train:dict, df_test:dict, n_layers:int, n_nodes:int, dropout:list, activation:str, optimizer:str, loss:str, metrics:list, epochs:int, batch_size:int,verbose:int)->object:
        """
        Neural network model to predict whether an item will be returned or not.

        Parameters
        ----------
        df_train : dict
            Dataframe with training data.
        df_test : dict
            Dataframe with testing data.
        n_layers : int
            Number of layers in the neural network.
        n_nodes : int
            Number of nodes in each layer.
        dropout : list
            List of dropout rates for each layer.
        activation : str
            Activation function for each layer except the last one.
        optimizer : str
            Optimizer for the neural network.
        loss : str
            Loss function for the neural network.
        metrics : str
            List of metrics for the neural network.
        epochs : int
            Number of epochs for the neural network.
        batch_size : int
            Size of the batch
        verbose : int
            Whether to print the progress of the neural network

        Returns
        -------
        model: object
            The Neural network model
        Y_pred: array
            Array of floats with the predictions of the neural network
        mae: float
            Mean absolute error on the testing set
        """         
        X_train, Y_train, X_test, Y_test = self.XY_split(df_train, df_test)
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
        
        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                model.add(Dense(n_nodes[i], input_dim=X_train.shape[1], activation=activation))
            else:
                model.add(Dense(n_nodes[i], activation=activation))
            model.add(Dropout(dropout[i]))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,verbose=verbose,validation_data=(X_test, Y_test))
        Y_pred = model.predict(X_test).round(4)
        mae = mean_absolute_error(Y_test, Y_pred)
        return model, Y_pred, mae,
    
    def xgboost(self, df_train:dict, df_test:dict, params:dict, verbose:int)->object, list, float,list:
        """ 
        XGBoost model to predict whether an item will be returned or not.

        Parameters
        ----------
        df_train : dict
            Dataframe with training data.
        df_test : dict
            Dataframe with testing data.
        params : dict
            Dictionary with the parameters for the XGBoost model.
        verbose : int
            Whether to print the progress of the XGBoost model.

        Returns
        -------
        model: object
            The XGBoost model
        Y_pred: array
            Array of floats with the predictions of the XGBoost model
        mae: float
            Mean absolute error on the testing set
        Y_pred_proba: list
            List of floats with the probabilities of the XGBoost model
        """
        X_train, Y_train, X_test, Y_test = self.XY_split(df_train, df_test)
        model = XGBClassifier(**params)
        model.fit(X_train, Y_train, eval_metric='mae', eval_set=[(X_test, Y_test)],early_stopping_rounds = 20,verbose = verbose)
        Y_pred = model.predict(X_test)
        Y_pred_proba = model.predict_proba(X_test)[:,1].round(4)
        mae = mean_absolute_error(Y_test, Y_pred)
        return model, Y_pred, mae, Y_pred_proba
    
    def catboost(self, df_train:dict, df_test:dict, params:dict, verbose:bool)->object, list, float,list:
        """ 
        CatBoost model to predict whether an item will be returned or not.

        Parameters
        ----------
        df_train : dict
            Dataframe with training data.
        df_test : dict
            Dataframe with testing data.
        params : dict
            Dictionary with the parameters for the CatBoost model.
        verbose : int
            Whether to print the progress of the CatBoost model.

        Returns
        -------
        model: object
            The CatBoost model
        Y_pred: array
            Array of floats with the predictions of the CatBoost model
        mae: float
            Mean absolute error on the testing set
        Y_pred_proba: list
            List of floats with the probabilities of the CatBoost model
        """
        X_train, Y_train, X_test, Y_test = self.XY_split(df_train, df_test)
        model = CatBoostClassifier(**params)
        model.fit(X_train, Y_train, eval_set=(X_test, Y_test), use_best_model=True, verbose=verbose,early_stopping_rounds = 20)
        Y_pred = model.predict(X_test)
        Y_pred_proba = model.predict_proba(X_test)[:,1].round(4)
        mae = mean_absolute_error(Y_test, Y_pred)
        return model, Y_pred, mae, Y_pred_proba
    
    def lightgmb(self, df_train:dict, df_test:dict, params:dict, verbose:int)->object, list, float,list:
        """ 
        LightGBM model to predict whether an item will be returned or not.

        Parameters
        ----------
        df_train : dict
            Dataframe with training data.
        df_test : dict
            Dataframe with testing data.
        params : dict
            Dictionary with the parameters for the LightGBM model.
        verbose : int
            Whether to print the progress of the LightGBM model.

        Returns
        -------
        model: object
            The LightGBM model
        Y_pred: array
            Array of floats with the predictions of the LightGBM model
        mae: float
            Mean absolute error on the testing set
        Y_pred_proba: list
            List of floats with the probabilities of the LightGBM model
        """
        X_train, Y_train, X_test, Y_test = self.XY_split(df_train, df_test)
        model = LGBMClassifier(**params)
        model.fit(X_train, Y_train, verbose=verbose)
        Y_pred_proba = model.predict_proba(X_test)[:,1].round(4)
        Y_pred = model.predict(X_test)
        mae = mean_absolute_error(Y_test, Y_pred)
        return model, Y_pred, mae, Y_pred_proba
    
    def combine_models(self, prob_dict,df_test:dict)->str,float:
        """ 
        Combines the predictions of the models to get the final prediction.

        Parameters
        ----------
        prob_dict : dict
            Dictionary with the probabilities of the models.
        df_test : dict
            Dataframe with testing data.

        Returns
        -------
        best_combination: str
            The name of the combination of models that gives the best result.
        best_error: float
            The mean absolute error of the combination of models that gives the best result.
        """
        prob_list = list(prob_dict.keys())
        best_error = 1
        for j in range(len(prob_dict)):
            for k in range(j+1,len(prob_dict)):
                for i in range(1,101):
                    first_term = prob_dict[prob_list[j]]*(i/100)
                    second_term = prob_dict[prob_list[k]]*((100-i)/100)
                    sum = first_term + second_term
                    error = mean_absolute_error(df_test["return"], sum.round())
                    if error < best_error:
                        best_error = error
                        best_i = i
                        best_model1 = prob_list[j] + "_" + str(i/100)
                        best_model2 = prob_list[k] + "_" + str((100-i)/100)
                        best_combination = best_model1 + "/" + best_model2
        return best_combination, best_error
    
    def XY_split(self, df_train:dict, df_test:dict)->dict,dict,list,list:
        """ 
        Splits the data into X and Y.

        Parameters
        ----------
        df_train : dict
            Dataframe with training data.
        df_test : dict
            Dataframe with testing data.

        Returns
        -------
        X_train: dict
            Dataframe with the features of the training data.
        Y_train: dict
            Dataframe with the labels of the training data.
        X_test: list
            List with the features of the testing data.
        Y_test: list
            List with the labels of the testing data.
        """
        X_train = df_train.drop(["return"], axis=1)
        Y_train = df_train["return"]
        X_test = df_test.drop(["return"], axis=1)
        Y_test = df_test["return"]
        return X_train, Y_train, X_test, Y_test
        X_train, Y_train = df_train.drop(["return"],axis=1), df_train["return"]
        X_test, Y_test = df_test.drop(["return"],axis=1), df_test["return"]
        return X_train, Y_train, X_test, Y_test
    
    
    def get_feature_importance(model:object,df:dict)->list:
        """
        Get feature importance from the tree-based models.

        Parameters
        ----------
        model : object
            Tree model from sklearn.
        df  : dict
            Tree model from sklearn

        Returns
        -------
        list
            List of the features with their corresponding feature importance.
        """    
        return list(sorted(zip(df.columns.drop("return"), model.feature_importances_), key=lambda xx: xx[1], reverse=True))
        

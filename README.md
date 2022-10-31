# Online-retailer-analysis
## Introduction
In this repository, You will find an example of online retailer analysis using python, with the goal of building a classifier to successfully predict if an ordered item is going to be returned or not. The dataset and the task of predicting the return of an item stem from the [Data Mining Cup 2014](https://www.data-mining-cup.com/reviews/dmc-2014/). The dataset encompasses 13 months of sales data for a German online clothing retailer and contains over `530.000 items` ordered. The dataset contains features describing the items, as well as the customer who placed the order.

Predicting if a customer is going to return an item is an important task for online retailers, as returned items incur costs for the retailer. For this reason, it could be beneficial if the retailer could predict if a customer is going to return an item before the item is shipped. This would allow the retailer to take action to prevent the return, such as limiting the number of items a customer can order, reminding the customer of the environmental impact of returning items, limiting his payment options, or even prohibiting the customer from ordering altogether.

The dataset has the following features:
- Order features
    - `order_id`: Index variable of the item.
    - `order_date`: Date of the order using the format `YYYY-MM-DD`.
    - `delivery_date`: Delivery date of the order using the format `YYYY-MM-DD`.
- Item features.
    - `item_id`: ID of the customer who ordered the item.
    - `item_size`: Size of the item.
    - `item_color`: Color of the item.
    - `brand_id`: ID of the brand of the item.
    - `item_price`: Price of the item.
- User features.
    - `user_id`:  ID of the user
    - `user_title`: Title of the user
    - `user_dob`: Date of birth of the user
    - `user_state`: German Federal state where the user lives
    - `user_reg_date`: Date of registration of the user
- Label
    - `return`: Label of the item. 0 if the item is not returned, and 1 if the item is returned.

## Feature Engineering
Overall we managed to generate over ``450 features`` using the base features. This was done both by using the base features directly, as well as by combining them with each other, or deriving new features from them, for example by using the date of birth of the user to calculate the age of the user or using the order and the delivery date to get the delivery time of the item.

## Modeling
The following models were used to predict the return of an item:
- `Neural Network`
- `XGBoost`
- `CatBoost`
- `LightGBM`

Since both the neural network and the XGBboost models do not natively support categorical features, and because the categorical features in the dataset possess a high cardinality, we used [Leave-One-Out(LOE)](https://contrib.scikit-learn.org/category_encoders/leaveoneout.html) encoding to encode the categorical features. This encoding method encodes each categorical feature by replacing each category with the mean of the target variable for the category while excluding the current row’s target when calculating the mean target for a level to reduce the effect of outliers.

## Results
After every round of feature engineering and modeling, we evaluated the performance of the models using the Mean Absolute Error (MAE) and removed models that did not perform well. Throughout several rounds of feature engineering and modeling as well as hyperparameter tuning, the best-performing model was the ``CatBoost`` model using its `GPU` implementation. Using the evaluation criteria defined in the Data Mining Cup 2014:
$$E= \sum_i^n \left\lvert returnShipment_i - prediction_i   \right\rvert$$

Where, $returnShipment_i$ is the information whether order item $i$ represents a return (0 means “item kept”, 1 means “item returned”), and $prediction_i$ is the predicted return probability for the order item $i$. The model achieved an MAE of `0.2974` on the test set and therefore ``14893 Points`` in the competition, which places the solution in the ``top 4`` of the [ranking list](https://www.wi.hs-wismar.de/~cleve/vorl/dmdaten/daten/DMC/DMC2014_Ranking.pdf) of the Data Mining Cup 2014. 

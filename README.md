# Online-retailer-analysis
## Introduction
In this repository, You will find an example of online retailer analysis using python, with the goal of predicting if an ordered item is going to be returned or not. The dataset and the task of predicting the return of an item stem from the Data Mining Cup 2014. The dataset encompasses 13 months of sales data for a German online clothing retailer and contains over 530 000 items ordered. The dataset contains features describing the items, as well as the customer who placed the order.

Predicting if a customer is going to return an item is an important task for online retailers, as returned items incur costs for the retailer. For this reason, it could be beneficial if the retailer could predict if a customer is going to return an item before the item is shipped. This would allow the retailer to take action to prevent the return, such as limiting the number of items a customer can order, reminding the customer of the environmental impact of returning items, limiting his payment options, or even prohibiting the customer from ordering altogether.

The dataset has the following 14 base features:
1. **"order_id"**: Index variable of the item.
2. **"order_date"**: Date of the order using the format **"YYYY-MM-DD"**.
3. **"delivery_date"**: Delivery date of the order using the format **"YYYY-MM-DD"**.
4. **"item_id"**: ID of the customer who ordered the item.
5. **"item_size"**: Size of the item.
6. **"item_color"**: Color of the item.
7. **"brand_id"**: ID of the brand of the item.
8. **"item_price"**: Price of the item.
9. **"user_id"**:  ID of the user
10. **"user_title"**: Title of the user
11. **"user_dob"**: Date of birth of the user
12. **"user_state"**: German state where the user lives
13. **"user_reg_date"**: Date of registration of the user
14. **"return"**: Label of the item. 0 if the item is not returned, 1 if the item is returned.

## Feature Engineering
Overall we managed to generate over 450 features using the base features. This was done both by using the base features directly, as well as by combining them with each other, or deriving new features from them, for example by using the date of birth of the user to calculate the age of the user or using the order and the delivery date to get the delivery time of the item.

## Models
The following models were used to predict the return of an item:
1. `ColorCode`
2. **XGBoost**
3. **Catboost**
4. **LightGBM**

Since both the neural network and the XGBboost models do not natively support categorical features, and because the categorical features in the dataset possess a high cardinality, we used [Leave-One-Out(LOE)](https://contrib.scikit-learn.org/category_encoders/leaveoneout.html) encoding to encode the categorical features. This encoding method encodes each categorical feature by replacing each category with the mean of the target variable for the category , while excluding the current row’s target when calculating the mean target for a level to reduce the effect of outliers.


# -*- coding: utf-8 -*-
"""
Loading in the data from the M5 forecasting competition https://www.kaggle.com/c/m5-forecasting-uncertainty/
In particular: sales_train_evaluation.csv (put in data/ folder)
Also reads in the forecast made during the accuracy stream, saved as submission_accuracy.csv

used in model.py

"""
import os
import pandas as pd

class M5Data:
    levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
    couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                            ("state_id", "cat_id"),("store_id","cat_id")]
    f_columns = [f"F{i}" for i in range(1, 29)]
    
    def __init__(self, data_path = None ):
        self.data_path = os.path.abspath("data") if data_path is None else data_path
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"There is no folder '{self.data_path}'.")
            
        self._sales_df = None
        self._acc_predictions = None
        
    def get_salesdf(self):
        if self._sales_df is None:
            self._sales_df = pd.read_csv(os.path.join(self.data_path, "sales_train_evaluation.csv"))
        return self._sales_df
    
    def get_accdf(self):
        if self._acc_predictions is None:
            self._acc_df = pd.read_csv(os.path.join(self.data_path, "submission_accuracy.csv"))
        return self._acc_df
    
    def get_merge_acc(self):
        self.sub_all = self._acc_df.merge(self._sales_df[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on = "id")
        self.sub_all["_all_"] = "Total"
        return self.sub_all
        
    
#%%


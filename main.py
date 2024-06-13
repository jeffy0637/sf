#import os
#os.chdir("C:/Users/jeffy/python/sf")

import functions_framework
import pandas as pd
import numpy as np
import torch
from google.cloud import storage

@functions_framework.http
def DRAGON_function(request):
    """
    Cloud Functions 處理函式

    Args:
        request: HTTP 請求物件，包含參數 user_ID

    Returns:
        JSON 格式的預測結果
    """

    import functions_framework

import pandas as pd
import torch
from google.cloud import storage

@functions_framework.http
def DRAGON_function(request):
    """
    Cloud Functions 處理函式

    Args:
        request: HTTP 請求物件，包含參數 user_ID

    Returns:
        JSON 格式的預測結果
    """

    # 取得 user_ID 參數
    request_json = request.get_json(silent=True)
    if request_json and 'user_ID' in request_json:
        user_ID = request_json['user_ID']
    else:
        return 'Error: Missing user_ID parameter', 400  # 返回錯誤訊息
    
    """
    # 從 GCS 載入模型
    storage_client = storage.Client()
    bucket_name = 'rec-modal'
    source_blob_name = 'active_modal/dragon_mg_outfit.pt'
    destination_file_name = '/tmp/dragon_mg_outfit.pt'  # 儲存到 Cloud Functions 的暫存空間

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
    except Exception as e:
        return f'Error downloading model: {e}', 500  # 返回錯誤訊息
    """
    
    try:
        model = torch.load('dragon_interupt.pt')
        #model = torch.load(destination_file_name)
    except Exception as e:
        return f'Error loading model: {e}', 500  # 返回錯誤訊息

    # 進行預測
    try:
        res = model.full_sort_predict_df([user_ID])
    except Exception as e:
        return f'Error during prediction: {e}', 500  # 返回錯誤訊息

    # 將 DataFrame 轉換為 JSON
    try:
        if isinstance(res, pd.DataFrame):
            res_json = res.to_json(orient='records')  # 將 DataFrame 轉換為 JSON 字串
        else:
            return 'Error: Prediction result is not a DataFrame', 500  # 返回錯誤訊息
    except Exception as e:
        return f'Error converting result to JSON: {e}', 500  # 返回錯誤訊息

    # 返回 JSON 結果
    return res_json, 200  # 200 表示成功

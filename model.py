import pickle as pkl
import uvicorn
def load_pickle_file(address):
  with open(address,'rb') as fil:
    return pkl.load(fil)



import warnings
warnings.filterwarnings("ignore")

def custom_encoder(var):
  out=0
  li=len(str(var))
  for i in range(li):
    out+=( ord(str(var)[i]) - ord('A')+1)*26**(li-i-1)
  return out

abc=['cat1',
 'cat2',
 'cat3',
 'cat4',
 'cat5',
 'cat6',
 'cat7',
 'cat8',
 'cat9',
 'cat10',
 'cat11',
 'cat12',
 'cat13',
 'cat14',
 'cat15',
 'cat16',
 'cat17',
 'cat18',
 'cat19',
 'cat20',
 'cat21',
 'cat22',
 'cat23',
 'cat24',
 'cat25',
 'cat26',
 'cat27',
 'cat28',
 'cat29',
 'cat30',
 'cat31',
 'cat32',
 'cat33',
 'cat34',
 'cat35',
 'cat36',
 'cat37',
 'cat38',
 'cat39',
 'cat40',
 'cat41',
 'cat42',
 'cat43',
 'cat44',
 'cat45',
 'cat46',
 'cat47',
 'cat48',
 'cat49',
 'cat50',
 'cat51',
 'cat52',
 'cat53',
 'cat54',
 'cat55',
 'cat56',
 'cat57',
 'cat58',
 'cat59',
 'cat60',
 'cat61',
 'cat62',
 'cat63',
 'cat64',
 'cat65',
 'cat66',
 'cat67',
 'cat68',
 'cat69',
 'cat70',
 'cat71',
 'cat72',
 'cat73',
 'cat74',
 'cat75',
 'cat76',
 'cat77',
 'cat78',
 'cat79',
 'cat80',
 'cat81',
 'cat82',
 'cat83',
 'cat84',
 'cat85',
 'cat86',
 'cat87',
 'cat88',
 'cat89',
 'cat90',
 'cat91',
 'cat92',
 'cat93',
 'cat94',
 'cat95',
 'cat96',
 'cat97',
 'cat98',
 'cat99',
 'cat100',
 'cat101',
 'cat102',
 'cat103',
 'cat104',
 'cat105',
 'cat106',
 'cat107',
 'cat108',
 'cat109',
 'cat110',
 'cat111',
 'cat112',
 'cat113',
 'cat114',
 'cat115',
 'cat116',
 'cont1',
 'cont2',
 'cont3',
 'cont4',
 'cont5',
 'cont6',
 'cont7',
 'cont8',
 'cont9',
 'cont10',
 'cont11',
 'cont12',
 'cont13',
 'cont14',
 'cat80_cat87',
 'cat80_cat57',
 'cat80_cat12',
 'cat80_cat79',
 'cat80_cat10',
 'cat80_cat7',
 'cat80_cat89',
 'cat80_cat2',
 'cat80_cat72',
 'cat80_cat81',
 'cat80_cat11',
 'cat80_cat1',
 'cat80_cat13',
 'cat80_cat9',
 'cat80_cat3',
 'cat80_cat16',
 'cat80_cat90',
 'cat80_cat23',
 'cat80_cat36',
 'cat80_cat73',
 'cat80_cat103',
 'cat80_cat40',
 'cat80_cat28',
 'cat80_cat111',
 'cat80_cat6',
 'cat80_cat76',
 'cat80_cat50',
 'cat80_cat5',
 'cat80_cat4',
 'cat80_cat14',
 'cat80_cat38',
 'cat80_cat24',
 'cat80_cat82',
 'cat80_cat25',
 'cat87_cat57',
 'cat87_cat12',
 'cat87_cat79',
 'cat87_cat10',
 'cat87_cat7',
 'cat87_cat89',
 'cat87_cat2',
 'cat87_cat72',
 'cat87_cat81',
 'cat87_cat11',
 'cat87_cat1',
 'cat87_cat13',
 'cat87_cat9',
 'cat87_cat3',
 'cat87_cat16',
 'cat87_cat90',
 'cat87_cat23',
 'cat87_cat36',
 'cat87_cat73',
 'cat87_cat103',
 'cat87_cat40',
 'cat87_cat28',
 'cat87_cat111',
 'cat87_cat6',
 'cat87_cat76',
 'cat87_cat50',
 'cat87_cat5',
 'cat87_cat4',
 'cat87_cat14',
 'cat87_cat38',
 'cat87_cat24',
 'cat87_cat82',
 'cat87_cat25',
 'cat57_cat12',
 'cat57_cat79',
 'cat57_cat10',
 'cat57_cat7',
 'cat57_cat89',
 'cat57_cat2',
 'cat57_cat72',
 'cat57_cat81',
 'cat57_cat11',
 'cat57_cat1',
 'cat57_cat13',
 'cat57_cat9',
 'cat57_cat3',
 'cat57_cat16',
 'cat57_cat90',
 'cat57_cat23',
 'cat57_cat36',
 'cat57_cat73',
 'cat57_cat103',
 'cat57_cat40',
 'cat57_cat28',
 'cat57_cat111',
 'cat57_cat6',
 'cat57_cat76',
 'cat57_cat50',
 'cat57_cat5',
 'cat57_cat4',
 'cat57_cat14',
 'cat57_cat38',
 'cat57_cat24',
 'cat57_cat82',
 'cat57_cat25',
 'cat12_cat79',
 'cat12_cat10',
 'cat12_cat7',
 'cat12_cat89',
 'cat12_cat2',
 'cat12_cat72',
 'cat12_cat81',
 'cat12_cat11',
 'cat12_cat1',
 'cat12_cat13',
 'cat12_cat9',
 'cat12_cat3',
 'cat12_cat16',
 'cat12_cat90',
 'cat12_cat23',
 'cat12_cat36',
 'cat12_cat73',
 'cat12_cat103',
 'cat12_cat40',
 'cat12_cat28',
 'cat12_cat111',
 'cat12_cat6',
 'cat12_cat76',
 'cat12_cat50',
 'cat12_cat5',
 'cat12_cat4',
 'cat12_cat14',
 'cat12_cat38',
 'cat12_cat24',
 'cat12_cat82',
 'cat12_cat25',
 'cat79_cat10',
 'cat79_cat7',
 'cat79_cat89',
 'cat79_cat2',
 'cat79_cat72',
 'cat79_cat81',
 'cat79_cat11',
 'cat79_cat1',
 'cat79_cat13',
 'cat79_cat9',
 'cat79_cat3',
 'cat79_cat16',
 'cat79_cat90',
 'cat79_cat23',
 'cat79_cat36',
 'cat79_cat73',
 'cat79_cat103',
 'cat79_cat40',
 'cat79_cat28',
 'cat79_cat111',
 'cat79_cat6',
 'cat79_cat76',
 'cat79_cat50',
 'cat79_cat5',
 'cat79_cat4',
 'cat79_cat14',
 'cat79_cat38',
 'cat79_cat24',
 'cat79_cat82',
 'cat79_cat25',
 'cat10_cat7',
 'cat10_cat89',
 'cat10_cat2',
 'cat10_cat72',
 'cat10_cat81',
 'cat10_cat11',
 'cat10_cat1',
 'cat10_cat13',
 'cat10_cat9',
 'cat10_cat3',
 'cat10_cat16',
 'cat10_cat90',
 'cat10_cat23',
 'cat10_cat36',
 'cat10_cat73',
 'cat10_cat103',
 'cat10_cat40',
 'cat10_cat28',
 'cat10_cat111',
 'cat10_cat6',
 'cat10_cat76',
 'cat10_cat50',
 'cat10_cat5',
 'cat10_cat4',
 'cat10_cat14',
 'cat10_cat38',
 'cat10_cat24',
 'cat10_cat82',
 'cat10_cat25',
 'cat7_cat89',
 'cat7_cat2',
 'cat7_cat72',
 'cat7_cat81',
 'cat7_cat11',
 'cat7_cat1',
 'cat7_cat13',
 'cat7_cat9',
 'cat7_cat3',
 'cat7_cat16',
 'cat7_cat90',
 'cat7_cat23',
 'cat7_cat36',
 'cat7_cat73',
 'cat7_cat103',
 'cat7_cat40',
 'cat7_cat28',
 'cat7_cat111',
 'cat7_cat6',
 'cat7_cat76',
 'cat7_cat50',
 'cat7_cat5',
 'cat7_cat4',
 'cat7_cat14',
 'cat7_cat38',
 'cat7_cat24',
 'cat7_cat82',
 'cat7_cat25',
 'cat89_cat2',
 'cat89_cat72',
 'cat89_cat81',
 'cat89_cat11',
 'cat89_cat1',
 'cat89_cat13',
 'cat89_cat9',
 'cat89_cat3',
 'cat89_cat16',
 'cat89_cat90',
 'cat89_cat23',
 'cat89_cat36',
 'cat89_cat73',
 'cat89_cat103',
 'cat89_cat40',
 'cat89_cat28',
 'cat89_cat111',
 'cat89_cat6',
 'cat89_cat76',
 'cat89_cat50',
 'cat89_cat5',
 'cat89_cat4',
 'cat89_cat14',
 'cat89_cat38',
 'cat89_cat24',
 'cat89_cat82',
 'cat89_cat25',
 'cat2_cat72',
 'cat2_cat81',
 'cat2_cat11',
 'cat2_cat1',
 'cat2_cat13',
 'cat2_cat9',
 'cat2_cat3',
 'cat2_cat16',
 'cat2_cat90',
 'cat2_cat23',
 'cat2_cat36',
 'cat2_cat73',
 'cat2_cat103',
 'cat2_cat40',
 'cat2_cat28',
 'cat2_cat111',
 'cat2_cat6',
 'cat2_cat76',
 'cat2_cat50',
 'cat2_cat5',
 'cat2_cat4',
 'cat2_cat14',
 'cat2_cat38',
 'cat2_cat24',
 'cat2_cat82',
 'cat2_cat25',
 'cat72_cat81',
 'cat72_cat11',
 'cat72_cat1',
 'cat72_cat13',
 'cat72_cat9',
 'cat72_cat3',
 'cat72_cat16',
 'cat72_cat90',
 'cat72_cat23',
 'cat72_cat36',
 'cat72_cat73',
 'cat72_cat103',
 'cat72_cat40',
 'cat72_cat28',
 'cat72_cat111',
 'cat72_cat6',
 'cat72_cat76',
 'cat72_cat50',
 'cat72_cat5',
 'cat72_cat4',
 'cat72_cat14',
 'cat72_cat38',
 'cat72_cat24',
 'cat72_cat82',
 'cat72_cat25',
 'cat81_cat11',
 'cat81_cat1',
 'cat81_cat13',
 'cat81_cat9',
 'cat81_cat3',
 'cat81_cat16',
 'cat81_cat90',
 'cat81_cat23',
 'cat81_cat36',
 'cat81_cat73',
 'cat81_cat103',
 'cat81_cat40',
 'cat81_cat28',
 'cat81_cat111',
 'cat81_cat6',
 'cat81_cat76',
 'cat81_cat50',
 'cat81_cat5',
 'cat81_cat4',
 'cat81_cat14',
 'cat81_cat38',
 'cat81_cat24',
 'cat81_cat82',
 'cat81_cat25',
 'cat11_cat1',
 'cat11_cat13',
 'cat11_cat9',
 'cat11_cat3',
 'cat11_cat16',
 'cat11_cat90',
 'cat11_cat23',
 'cat11_cat36',
 'cat11_cat73',
 'cat11_cat103',
 'cat11_cat40',
 'cat11_cat28',
 'cat11_cat111',
 'cat11_cat6',
 'cat11_cat76',
 'cat11_cat50',
 'cat11_cat5',
 'cat11_cat4',
 'cat11_cat14',
 'cat11_cat38',
 'cat11_cat24',
 'cat11_cat82',
 'cat11_cat25',
 'cat1_cat13',
 'cat1_cat9',
 'cat1_cat3',
 'cat1_cat16',
 'cat1_cat90',
 'cat1_cat23',
 'cat1_cat36',
 'cat1_cat73',
 'cat1_cat103',
 'cat1_cat40',
 'cat1_cat28',
 'cat1_cat111',
 'cat1_cat6',
 'cat1_cat76',
 'cat1_cat50',
 'cat1_cat5',
 'cat1_cat4',
 'cat1_cat14',
 'cat1_cat38',
 'cat1_cat24',
 'cat1_cat82',
 'cat1_cat25',
 'cat13_cat9',
 'cat13_cat3',
 'cat13_cat16',
 'cat13_cat90',
 'cat13_cat23',
 'cat13_cat36',
 'cat13_cat73',
 'cat13_cat103',
 'cat13_cat40',
 'cat13_cat28',
 'cat13_cat111',
 'cat13_cat6',
 'cat13_cat76',
 'cat13_cat50',
 'cat13_cat5',
 'cat13_cat4',
 'cat13_cat14',
 'cat13_cat38',
 'cat13_cat24',
 'cat13_cat82',
 'cat13_cat25',
 'cat9_cat3',
 'cat9_cat16',
 'cat9_cat90',
 'cat9_cat23',
 'cat9_cat36',
 'cat9_cat73',
 'cat9_cat103',
 'cat9_cat40',
 'cat9_cat28',
 'cat9_cat111',
 'cat9_cat6',
 'cat9_cat76',
 'cat9_cat50',
 'cat9_cat5',
 'cat9_cat4',
 'cat9_cat14',
 'cat9_cat38',
 'cat9_cat24',
 'cat9_cat82',
 'cat9_cat25',
 'cat3_cat16',
 'cat3_cat90',
 'cat3_cat23',
 'cat3_cat36',
 'cat3_cat73',
 'cat3_cat103',
 'cat3_cat40',
 'cat3_cat28',
 'cat3_cat111',
 'cat3_cat6',
 'cat3_cat76',
 'cat3_cat50',
 'cat3_cat5',
 'cat3_cat4',
 'cat3_cat14',
 'cat3_cat38',
 'cat3_cat24',
 'cat3_cat82',
 'cat3_cat25',
 'cat16_cat90',
 'cat16_cat23',
 'cat16_cat36',
 'cat16_cat73',
 'cat16_cat103',
 'cat16_cat40',
 'cat16_cat28',
 'cat16_cat111',
 'cat16_cat6',
 'cat16_cat76',
 'cat16_cat50',
 'cat16_cat5',
 'cat16_cat4',
 'cat16_cat14',
 'cat16_cat38',
 'cat16_cat24',
 'cat16_cat82',
 'cat16_cat25',
 'cat90_cat23',
 'cat90_cat36',
 'cat90_cat73',
 'cat90_cat103',
 'cat90_cat40',
 'cat90_cat28',
 'cat90_cat111',
 'cat90_cat6',
 'cat90_cat76',
 'cat90_cat50',
 'cat90_cat5',
 'cat90_cat4',
 'cat90_cat14',
 'cat90_cat38',
 'cat90_cat24',
 'cat90_cat82',
 'cat90_cat25',
 'cat23_cat36',
 'cat23_cat73',
 'cat23_cat103',
 'cat23_cat40',
 'cat23_cat28',
 'cat23_cat111',
 'cat23_cat6',
 'cat23_cat76',
 'cat23_cat50',
 'cat23_cat5',
 'cat23_cat4',
 'cat23_cat14',
 'cat23_cat38',
 'cat23_cat24',
 'cat23_cat82',
 'cat23_cat25',
 'cat36_cat73',
 'cat36_cat103',
 'cat36_cat40',
 'cat36_cat28',
 'cat36_cat111',
 'cat36_cat6',
 'cat36_cat76',
 'cat36_cat50',
 'cat36_cat5',
 'cat36_cat4',
 'cat36_cat14',
 'cat36_cat38',
 'cat36_cat24',
 'cat36_cat82',
 'cat36_cat25',
 'cat73_cat103',
 'cat73_cat40',
 'cat73_cat28',
 'cat73_cat111',
 'cat73_cat6',
 'cat73_cat76',
 'cat73_cat50',
 'cat73_cat5',
 'cat73_cat4',
 'cat73_cat14',
 'cat73_cat38',
 'cat73_cat24',
 'cat73_cat82',
 'cat73_cat25',
 'cat103_cat40',
 'cat103_cat28',
 'cat103_cat111',
 'cat103_cat6',
 'cat103_cat76',
 'cat103_cat50',
 'cat103_cat5',
 'cat103_cat4',
 'cat103_cat14',
 'cat103_cat38',
 'cat103_cat24',
 'cat103_cat82',
 'cat103_cat25',
 'cat40_cat28',
 'cat40_cat111',
 'cat40_cat6',
 'cat40_cat76',
 'cat40_cat50',
 'cat40_cat5',
 'cat40_cat4',
 'cat40_cat14',
 'cat40_cat38',
 'cat40_cat24',
 'cat40_cat82',
 'cat40_cat25',
 'cat28_cat111',
 'cat28_cat6',
 'cat28_cat76',
 'cat28_cat50',
 'cat28_cat5',
 'cat28_cat4',
 'cat28_cat14',
 'cat28_cat38',
 'cat28_cat24',
 'cat28_cat82',
 'cat28_cat25',
 'cat111_cat6',
 'cat111_cat76',
 'cat111_cat50',
 'cat111_cat5',
 'cat111_cat4',
 'cat111_cat14',
 'cat111_cat38',
 'cat111_cat24',
 'cat111_cat82',
 'cat111_cat25',
 'cat6_cat76',
 'cat6_cat50',
 'cat6_cat5',
 'cat6_cat4',
 'cat6_cat14',
 'cat6_cat38',
 'cat6_cat24',
 'cat6_cat82',
 'cat6_cat25',
 'cat76_cat50',
 'cat76_cat5',
 'cat76_cat4',
 'cat76_cat14',
 'cat76_cat38',
 'cat76_cat24',
 'cat76_cat82',
 'cat76_cat25',
 'cat50_cat5',
 'cat50_cat4',
 'cat50_cat14',
 'cat50_cat38',
 'cat50_cat24',
 'cat50_cat82',
 'cat50_cat25',
 'cat5_cat4',
 'cat5_cat14',
 'cat5_cat38',
 'cat5_cat24',
 'cat5_cat82',
 'cat5_cat25',
 'cat4_cat14',
 'cat4_cat38',
 'cat4_cat24',
 'cat4_cat82',
 'cat4_cat25',
 'cat14_cat38',
 'cat14_cat24',
 'cat14_cat82',
 'cat14_cat25',
 'cat38_cat24',
 'cat38_cat82',
 'cat38_cat25',
 'cat24_cat82',
 'cat24_cat25',
 'cat82_cat25']

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import skew,boxcox
from pydantic import BaseModel
from typing import Dict
from urllib.parse import unquote
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

class Data(BaseModel):
  cat1:str
  cat2:str
  cat3:str
  cat4:str
  cat5:str
  cat6:str
  cat7:str
  cat8:str
  cat9:str
  cat10:str
  cat11:str
  cat12:str
  cat13:str
  cat14:str
  cat15:str
  cat16:str
  cat17:str
  cat18:str
  cat19:str
  cat20:str
  cat21:str
  cat22:str
  cat23:str
  cat24:str
  cat25: str
  cat26: str
  cat27: str
  cat28: str
  cat29: str
  cat30: str
  cat31: str
  cat32: str
  cat33: str
  cat34: str
  cat35: str
  cat36: str
  cat37: str
  cat38: str
  cat39: str
  cat40: str
  cat41: str
  cat42: str
  cat43: str
  cat44: str
  cat45: str
  cat46: str
  cat47: str
  cat48: str
  cat49: str
  cat50: str
  cat51: str
  cat52: str
  cat53: str
  cat54: str
  cat55: str
  cat56: str
  cat57: str
  cat58: str
  cat59: str
  cat60: str
  cat61: str
  cat62: str
  cat63: str
  cat64: str
  cat65: str
  cat66: str
  cat67: str
  cat68: str
  cat69: str
  cat70: str
  cat71: str
  cat72: str
  cat73: str
  cat74: str
  cat75: str
  cat76: str
  cat77: str
  cat78: str
  cat79: str
  cat80: str
  cat81: str
  cat82: str
  cat83: str
  cat84: str
  cat85: str
  cat86: str
  cat87: str
  cat88: str
  cat89: str
  cat90: str
  cat91: str
  cat92: str
  cat93: str
  cat94: str
  cat95: str
  cat96: str
  cat97: str
  cat98: str
  cat99: str
  cat100: str
  cat101: str
  cat102: str
  cat103: str
  cat104: str
  cat105: str
  cat106: str
  cat107: str
  cat108: str
  cat109: str
  cat110: str
  cat111: str
  cat112: str
  cat113: str
  cat114: str
  cat115: str
  cat116: str
  cont1: float
  cont2: float
  cont3: float
  cont4: float
  cont5: float
  cont6: float
  cont7: float
  cont8: float
  cont9: float
  cont10: float
  cont11: float
  cont12: float
  cont13: float
  cont14: float

from fastapi import FastAPI
final_api = FastAPI()
import nest_asyncio
nest_asyncio.apply()

@final_api.get('/')
async def front_page():
  return 'input the data'

@final_api.post('/predict/')
async def predict_the_claim(data:Data):
  cat=[]
  data=data.dict()
  for i in range(116):
    cat.append('cat'+str(i+1))
  data=pd.DataFrame(data,index=[0])
  scaler=load_pickle_file('scaler.pkl')
  conts=['cont'+str(i) for i in range(1,15)]
  for i in conts:
    data[i]=scaler[i].transform(data[i].values.reshape(-1,1))
  s=['cont1', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10',
       'cont11', 'cont12', 'cont13', 'cont14']
  for i in s:
    data[i]=data[i]+1
    data[i]=np.log(data[i])
  COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,' \
               'cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,' \
               'cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,' \
               'cat4,cat14,cat38,cat24,cat82,cat25'.split(',')
  
  for i in range(len(COMB_FEATURE)):
      for j in range(i+1,len(COMB_FEATURE)):
          feature = COMB_FEATURE[i]+'_'+COMB_FEATURE[j]
          data[feature]=data[COMB_FEATURE[i]]+data[COMB_FEATURE[j]]
          data[feature]=data[feature].apply(custom_encoder)
  for i in cat:
    data[i]=data[i].apply(custom_encoder)
  data=data[abc]
  prediction=0
  for i in range(10):
    clf= load_pickle_file('model'+str(i)+'.pkl')
    prediction=prediction+np.exp(clf.predict(xgb.DMatrix(data)))-200
  prediction=prediction/10
  output={'output':str(prediction[0])}
  output=jsonable_encoder(output)
  return  JSONResponse(content=output)

if __name__ == "__main__":
    uvicorn.run(final_api, host="localhost", port=80)

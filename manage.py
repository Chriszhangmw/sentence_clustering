
from apps.cal_cluster import AffinityPropagation
import pandas as pd
from sanic import Sanic
from sanic.response import json,text

app = Sanic(__name__)

@app.route('/sanic/vi/cluster',methods=['POST'])
def get_simi(request):
    data = request.json
    text_list_json = data['text_list_json']
    if not text_list_json:
        return {"code":208,"message":"提交内容为空"}
    pd_data = pd.DataFrame(text_list_json)
    if pd_data["text_id"].isnull().any():
        return {"code":208,"message":"text_id有误"}

    ap = AffinityPropagation()
    results = ap.get_result_clustering_by_ap(text_list_json)
    return text({"code":100,"message":"返回正常","data":results})

if __name__ == '__main__':
    app.run("0.0.0.0",port=5002)





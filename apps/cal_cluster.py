


import numpy as np
from sklearn.cluster import AffinityPropagation as AP
import config

import  jieba
from data.build_vec import load_w2v,load_word_fre

w2v = load_w2v(config.w2v_path)
word_freq = load_word_fre(config.fre_path)

max_clusters_sim = config.max_clustering_sim
ap_dampling = config.ap_damping


class  Cluster():
    def get_cos_similarity(self,v1,v2):
        vector_a = np.mat(v1)
        vector_b = np.mat(v2)

        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

        cos = num /denom

        sim = 0.5 + 0.5 * cos

        return sim

    def get_data_sims(self,x_n):
        text_sim_list = []
        for m in x_n:
            m = np.array(m)
            temps = []
            for n in x_n:
                n = np.array(n)
                sim = self.get_cos_similarity(m,n)
                temps.append(sim)
            text_sim_list.append(temps)

        x_sims = np.array(text_sim_list)
        max_sim = np.max(x_sims)
        min_sim = np.min(x_sims)
        mid_sim = np.median(x_sims)

        return x_sims,max_sim,min_sim,mid_sim


    def get_text_data(self,sent_list):
        a = 0.001
        row = w2v.wv.vector_size
        col = len(sent_list)

        sent_mat = np.mat(np.zeros((col,row)))
        text_vector_list = []
        for i,sent in enumerate(sent_list):
            text_id = sent_list[i].get('text_id')
            text_content = sent_list[i].get('text_content')
            new_sent = list(jieba.cut(str(text_content).strip()))
            if not new_sent:continue
            sent_vec = np.zeros(row)
            for word in new_sent:
                pw = word_freq.get(word,100)
                w = a / (a + pw)
                try:
                    vec = np.array(w2v.wv[word])
                    sent_vec += w * vec
                except:
                    pass
            text_vector_list.append({"text_id":text_id,"text_vector":sent_vec})
            sent_mat[i,:] += np.mat(sent_vec)
            sent_mat[i,:] /= len(new_sent)

        return sent_mat,text_vector_list

class AffinityPropagation(object):
    def get_result_clustering_by_ap(self,text_list):
        cluster = Cluster()
        Xn, text_vector_list = cluster.get_text_data(text_list)

        x_sims, max_sim, min_sim, mid_sim = cluster.get_data_sims(Xn)

        text_clustering_list = []
        if min_sim > max_clusters_sim:
            for i in range(len(text_vector_list)):
                text_id = text_vector_list[i].get("text_id")
                text_clustering_list.append({"text_id":text_id,"class_num":"0"})
        else:
            ap = AP(ap_dampling=ap_dampling,
                    max_iter=1000,
                    convergence_iter=100,
                    preference=mid_sim,
                    affinity="precomputed").fit(x_sims)
            labels = ap.labels_
            monitor_set = set()
            if -1 not in labels:
                for i in range(len(labels)):
                    monitor_set.add(labels[i])
                    class_num = str(labels[i])
                    text_id = text_vector_list[i].get("text_id")
                    text_clustering_list.append({"text_id":text_id,"class_num":class_num})
                return text_clustering_list
            else:
                adapt_ap_damping = 0.5
                ap = AP(ap_dampling=ap_dampling,
                    max_iter=1000,
                    convergence_iter=100,
                    preference=mid_sim,
                    affinity="precomputed").fit(x_sims)
                for i in range(len(labels)):
                    monitor_set.add(labels[i])
                    class_num = str(labels[i])
                    text_id = text_vector_list[i].get("text_id")
                    text_clustering_list.append({"text_id":text_id,"class_num":class_num})
                return text_clustering_list




if __name__ == "__main__":
    text_list = [
        {
            "text_id":"123",
            "text_content":"本次发病以来。神志清，精神差，睡眠可，大小便正常"
        },
        {
            "text_id":"1234",
            "text_content":"本次发病以来。神志清，精神差，睡眠可，大小便正常"
        },
        {
            "text_id": "1235",
            "text_content": "本次发病以来。神志清，精神差，睡眠可，大小便正常"
        },
        {
            "text_id": "1236",
            "text_content": "本次发病以来。神志清，精神差，睡眠可，大小便正常"
        },
        {
            "text_id": "1237",
            "text_content": "现病史：患者半月前无明显诱因出现胸疼，气喘不适，休息后可自行缓解，伴恶心呕吐不适，未给予特殊治疗，今来我院门诊，血常规"
        }
    ]






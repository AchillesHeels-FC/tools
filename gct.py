# coding=utf-8
import math
import warnings
import itertools
import community
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from configparser import ConfigParser
from networkx.drawing.nx_pydot import write_dot
from networkx import edge_betweenness_centrality as betweenness
warnings.filterwarnings("ignore")


class GraphCalculate(object):
    def __init__(self, conf_file):
        self.conf_file = conf_file
        self.cfg = ConfigParser()
        self.cfg.read(self.conf_file)
        self.sample_path = self.cfg.get('sample_path', 'sample_path')
        self.sample = pd.read_csv(self.sample_path, self.cfg.get('sep_string', 'sep'), header=0)
        self.multi_bool = None
        self.graph_dict = defaultdict()
        self.community_dict = defaultdict(list)
        self.community_scores = defaultdict(list)

    def distinct(self, dataframe):
        """
        去除重复值
        """
        df = dataframe.copy()
        df.drop_duplicates(inplace=True)  # 去除重复值
        df.index = range(df.shape[0])  # 去重后恢复索引
        return df

    def cal_jaccard(self, start_rc, end_rc):
        """
        通过 jaccard 相似性计算边权重
        """
        a = start_rc & end_rc
        b = start_rc | end_rc
        weight_edge = round(float(len(a)) / len(b), 2)
        return weight_edge

    def build_relation(self, base_dataframe):
        """
        构建关联数据表
        """
        df = base_dataframe.copy()
        df = df.astype('str')
        relate_condition = self.cfg.get('edge', 'relation')
        relate_condition = relate_condition.strip().split(',')
        node_name = self.cfg.get('node', 'node_dimension')
        weight_cal_func = self.cfg.get('weight', 'weight_cal_func')
        weight_col = self.cfg.get('weight', 'weight_col')
        weight_setting = self.cfg.get('weight', 'weight_setting')
        label_col = self.cfg.get('label', 'label_col')
        relate_data = []
        relate_colnames = ['start_node', 'end_node', 'relation', 'attribute', 'weight', 'start_label', 'end_label']
        for rc in relate_condition:
            for A, B in itertools.combinations(list(df[node_name]), 2):
                A_rc = df.loc[df.loc[:, node_name] == A, rc].values[0]
                B_rc = df.loc[df.loc[:, node_name] == B, rc].values[0]
                A_rc = set(str(A_rc).strip().split(','))
                B_rc = set(str(B_rc).strip().split(','))
                if A_rc.intersection(B_rc):
                    start_node = A
                    end_node = B
                    relation = ','.join(B_rc.intersection(B_rc))
                    attribute = rc
                    start_label = int(df.loc[df.loc[:, node_name] == A, label_col])
                    end_label = int(df.loc[df.loc[:, node_name] == B, label_col])
                    weight = 0
                    if weight_cal_func == 'default':
                        start_weight = float(df.loc[df.loc[:, node_name] == A, weight_col])
                        end_weight = float(df.loc[df.loc[:, node_name] == B, weight_col])
                        weight = max(start_weight, end_weight) if weight_setting == 'max' \
                            else min(start_weight, end_weight)
                    elif weight_cal_func == 'jaccard':
                        weight = self.cal_jaccard(A_rc, B_rc)
                    relate_data.append([start_node, end_node, relation, attribute, weight, start_label, end_label])
        relate_dfs = pd.DataFrame(relate_data, columns=relate_colnames, index=None)
        return relate_dfs

    def build_graph(self, dataframe):
        """
        构建图数据
        """
        def __build__(df, g_type):
            """
            构图内置方法
            """
            _df = df.copy()
            data = []
            edge_color = ['tomato', 'aqua', 'deeppink', 'yellow', 'k', 'lime']
            edge_style = ['solid', 'dashed', 'dashdot', 'dotted']
            nodes_pair_dict = defaultdict(int)
            graph = getattr(nx, g_type)()
            for row_index, row in _df.iterrows():
                s_n = row['start_node']
                e_n = row['end_node']
                weight = row['weight']
                relation = row['relation']
                s_l = row['start_label']
                e_l = row['end_label']
                nodes_pair_dict[(s_n, e_n)] += 1
                data.append((s_n, e_n, {'weight': weight, 'relation': relation, 's_l': s_l, 'e_l': e_l,
                                                    'edge_color': edge_color[nodes_pair_dict[(s_n, e_n)]-1],
                                                    'edge_style': edge_style[nodes_pair_dict[(s_n, e_n)]-1]}))
            graph.add_edges_from(data)
            return graph

        df = dataframe.copy()
        attr_discount = df.attribute.nunique()
        attrs = list(df['attribute'].unique())
        if attr_discount > 1:
            print('Hello old Baby! You now have many relationships: %s' % (','.join(attrs)))
            self.multi_bool = input('Whether you want to build a MultiGraph ? (Y/N): ')
        if self.multi_bool not in ('y', 'Y', 'yes', 'Yes'):
            g_type = self.cfg.get('Graph', '1')
            for attr in attrs:
                print('You now will build graph base %s relationship' % (attr))
                attr_df = df.loc[df['attribute'] == attr, :]
                self.graph_dict[attr] = __build__(attr_df, g_type)
        else:
            attr = '_'.join(attrs)
            g_type = self.cfg.get('Graph', '2')
            self.graph_dict[attr] = __build__(df, g_type)

    def show_relate_graph(self):
        """
        基础关联图可视化
        """
        graph_count = len(self.graph_dict)
        node_attr_flag = self.cfg.get('node', 'node_attribute_flag')
        edge_attr_flag = self.cfg.get('edge', 'edge_attribute_flag')
        edge_attr_show = self.cfg.get('edge', 'edge_attr_show').strip().split(',')
        node_attr_show = self.cfg.get('node', 'node_attr_show').strip().split(',')
        node_name = self.cfg.get('node', 'node_dimension')
        base_data = self.sample
        pos_func = self.cfg.get('layout', 'pos')
        plt.figure(figsize=(35, 15))
        plt.axis('off')
        index = 0
        for attr in self.graph_dict:
            graph = self.graph_dict[attr]
            colors = [attr_dict['edge_color'] for (s, e, attr_dict) in list(graph.edges.data())]
            styles = [attr_dict['edge_style'] for (s, e, attr_dict) in list(graph.edges.data())]
            pos = getattr(nx, pos_func)(graph)
            plt.subplot(2, graph_count / 2 + 1, index + 1)
            plt.title('base < %s > graph' % (attr))
            nx.draw(graph, pos=pos, with_labels=True, node_size=150, node_color='green',
                    alpha=0.7, edge_color=colors, width=0.7, style=styles, node_shape='o')
            if node_attr_flag == 'True':
                node_labels = defaultdict(dict)
                for node in graph.nodes:
                    for n_attr in node_attr_show:
                        node_labels[node][n_attr] = base_data.loc[base_data.loc[:, node_name] == node, n_attr].values[0]
                self.node_labels = node_labels
                nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=5)
            if edge_attr_flag == 'True':
                edge_labels = defaultdict(dict)
                for edge in graph.edges:
                    for e_attr in edge_attr_show:
                        edge_labels[edge][e_attr] = graph[edge[0]][edge[1]][e_attr]
                self.edge_labels = edge_labels
                nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)
            index += 1
        plt.show()

    def detect_community(self):
        """
        利用图计算进行社区发现
        """
        def most_central_edge(graph):
            """
            中心度加权
            """
            centrality = betweenness(graph, weight="weight")
            return max(centrality, key=centrality.get)

        community_func = self.cfg.get('Community', 'community')
        pos_func = self.cfg.get('Community', 'community_pos')
        edge_w_t = float(self.cfg.get('weight', 'edge_weight_threshold'))
        graphs = self.graph_dict
        for attr in graphs:
            graph = graphs[attr]
            pos = getattr(nx, pos_func)(graph)
            if self.multi_bool in ('y', 'Y', 'yes', 'Yes'):
                edge_weight_dict = defaultdict(float)
                for (s, e, attr_dict) in list(graph.edges.data()):
                    edge_weight_dict[(s, e)] += float(attr_dict['weight'])
                data = [(node_pair[0], node_pair[1], edge_weight_dict[node_pair]) for node_pair in edge_weight_dict]
                graph = nx.Graph()
                graph.add_weighted_edges_from(data)
            low_weight_edges = [(edge[0], edge[1]) for edge in graph.edges if
                                float(graph[edge[0]][edge[1]]['weight']) <= edge_w_t]
            graph.remove_edges_from(low_weight_edges)
            plt.figure(figsize=(25, 8))
            plt.axis('off')
            plt.title('community detect < %s > graph' % (attr))
            community_data = []
            count = 0
            node_colors = sns.hls_palette(n_colors=10, l=.7, s=.9)
            if community_func == 'best_partition':
                print('You will use Louvain algorithms to detect community by %s relationships' % (attr))
                partition = community.best_partition(graph, weight='weight')
                size = len(set(partition.values()))
                node_colors = node_colors * int(size)
                for com in set(partition.values()):
                    nodes = tuple(nodes for nodes in partition.keys() if partition[nodes] == com)
                    community_data.append(nodes)
                    nx.draw_networkx_nodes(graph, pos, nodes, node_size=100, node_color=[node_colors[count]]*len(nodes))
                    count += 1
            elif community_func == 'girvan_newman':
                print('You will use GN to detect community by %s relationships' % (attr))
                partition = nx.algorithms.community.girvan_newman(graph, most_valuable_edge=most_central_edge)  # 返回所有社团的组合
                community_count = int(input("Please enter the number of clubs you would like to discover: "))
                node_colors = node_colors * community_count
                limited = itertools.takewhile(lambda x: len(x) <= community_count, partition)
                for communities in limited:
                    if len(communities) == community_count:
                        community_data = list(map(lambda x: tuple(x), list(communities)))
                        for com in community_data:
                            nx.draw_networkx_nodes(graph, pos, list(com), node_size=100,
                                                   node_color=[node_colors[count]]*len(com))
                            count += 1
                        break
            nx.draw_networkx_edges(graph, pos, alpha=0.5, edge_color='black', style='dashdot')
            self.community_dict[attr] = community_data
            plt.show()

    def community_scoring(self):
        """
        计算社区欺诈分数
        """
        graphs = self.graph_dict
        for attr in graphs:
            graph = graphs[attr]
            graph = graph.to_undirected()
            communities = self.community_dict[attr]
            for com in communities:
                weights = []
                spams = set()
                node_count = len(com)
                if len(com) <= 1:
                    continue
                for s, e in itertools.combinations(list(com), 2):
                    try:
                        weights.append(float(graph[s][e]['weight']))
                        if graph[s][e]['s_l'] == 1:
                            spams.add(s)
                        if graph[s][e]['e_l'] == 1:
                            spams.add(e)
                    except Exception as e:
                        pass
                score = np.mean(weights) * (1 + math.log(node_count, 2)) * (1 + len(spams) / float(node_count))
                self.community_scores[attr].append([com, score])

    def output_community_info(self):
        """
        社团挖掘结果输出
        """
        for attr, com_scores in self.community_scores.items():
            print('base %s detect community and scores: ' % (attr))
            for c_s in com_scores:
                c = c_s[0]
                s = c_s[1]
                print(','.join(list(c)) + '\t' + str(s))

    def run(self, relate_data):
        """
        图计算执行入口
        """
        self.build_graph(relate_data)
        self.show_relate_graph()
        self.detect_community()
        self.community_scoring()
        self.output_community_info()


def main():
    gc = GraphCalculate(r'./gct.conf')
    base_data = gc.sample
    base_data = gc.distinct(base_data)
    relate_dfs = gc.build_relation(base_data)
    gc.run(relate_dfs)


if __name__ == '__main__':
    Begin_title = r"""

    ____  __                                __   _____       __    ______                 __  
   / __ \/ /___ ___  __   _________  ____  / /  / ___/__  __/ /_  / ____/________ _____  / /_ 
  / /_/ / / __ `/ / / /  / ___/ __ \/ __ \/ /   \__ \/ / / / __ \/ / __/ ___/ __ `/ __ \/ __ \
 / ____/ / /_/ / /_/ /  / /__/ /_/ / /_/ / /   ___/ / /_/ / /_/ / /_/ / /  / /_/ / /_/ / / / /
/_/   /_/\__,_/\__, /   \___/\____/\____/_/   /____/\__,_/_.___/\____/_/   \__,_/ .___/_/ /_/ 
              /____/                                                           /_/            

    """
    Begin_pic = """         ¸ ¸ ¸ ¸ ¸¸ ¸  ¸ 
               ;¸'::::::::::::::::::::':::·   .  .·'·¸ 
                  ` · :¸::::::::::,··,::::::::::` ··'. 
                          ` · .:::'·.·'::::::::::::::::`·. 
                 ¸  . .. · :: ·.::::::::::::¸:::::::::::::'. 
            ¸. '¸::::::::::::::::::::::.'     '·,::::::::::¸·  
                   '  ·  .:::::::::::;' .·.      ',::::::·' ;   
                      .  · ':::::::::; ';:';       ·:::;¸   '.  
                  . ·'·::.:::::::¸::¸'. '..'        '.';  ;   '. 
        .¸. · '::. · '·. ·:::::::'.    ` .          ' ;'·'   .' 
   . · '· .::.·';.   ·':::::::::::'`· .`·. `  · · ··¸.'¸¸.·'  
¸'  ¸      .'   '·...:::::. · ´ ` ·:.:.` ·`´ . ¸  .·'···'    
 ' ;.·;.·'      :..::::.'            ''.' ·¸               
                  .'   '·.           .'..·¸ ' ·.         ¸,¸....¸ 
                .'    .' · .`.· . · '· · ·. `·. `¸. ·;' . ':¸.¸ .;  
              .'    .'                   .'    ' .:  '  ` ·'..¸.¸: 
          ¸. '   ¸.'               .· . '     .'   '' · ....'·..·' 
      . '. ' · '.·             .·'·.  '·. · '*RL-ones* 
  . '. · ·'.·.'            . '. · . `..· ´  ' *MSøA* 
.'.  ·  · ..'           · .  · . .·'           *97' *    
'·.¸¸ . · '            ' .¸ . · '                 *«
    """
    print(Begin_title)
    print(Begin_pic)
    main()

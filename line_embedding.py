
from cgi import test
from re import A
from model.line import LINE
import torch
import networkx as nx
import numpy as np
print(torch.cuda.is_available())


def get_graph_0():
    G = nx.Graph()
    all_node = []
    all_edge = []
    with open("./data/gmd.edgelist") as f_edge:
        edge_content = f_edge.readlines()
        for edge in edge_content:
            node_1 = edge.strip("\n").split("\t")[0]
            node_2 = edge.strip("\n").split("\t")[1]
            weight = float(edge.strip("\n").split("\t")[2])
            if weight >0:
                all_edge.append((node_1, node_2))
            if node_1 not in all_node:
                all_node.append(node_1)
            if node_2 not in all_node:
                all_node.append(node_2)
    G.add_nodes_from(all_node)
    G.add_edges_from(all_edge)

    print("节点数量：", G.number_of_nodes())
    print("边的数量：", G.number_of_edges())

    return G

# def get_graph():    # 根据边文件生成图，用于节点嵌入
#     print("开始根据边文件生成Graph")
#     G = nx.Graph()

#     node_list = list(range(4677))
#     G.add_nodes_from(node_list) # 首先在图中添加节点，然后再连接边

#     with open("../0_data/edge_weight.txt") as f_edge:
#         edge_content = f_edge.readlines()
#         edge_list = []
#         for edge in edge_content:
#             head = int(edge.strip("\n").split(" ")[0])
#             tail = int(edge.strip("\n").split(" ")[1])
#             weight = float(edge.strip("\n").split(" ")[2])

#             htw = (head, tail, weight)
#             edge_list.append(htw)
#         G.add_weighted_edges_from(edge_list)
#     print("节点数量：", G.number_of_nodes())
#     print("边的数量：", G.number_of_edges())
#     return G

# 读取训练集测试集样本
filea = open("data/X_train.txt")
x_train = filea.readlines()
fileb = open("data/y_train.txt")
y_train = fileb.readlines()

filea = open("data/X_test.txt")
X_test = filea.readlines()
fileb = open("data/y_test.txt")
y_test = fileb.readlines()
# 训练集
train = []
for i in range(len(x_train)):
        tup1 = (x_train[i].split()[0].split(",")[0],)
        tup2 = (x_train[i].split()[0].split(",")[1],)
        tup3 = (y_train[i].split()[0],)
        tup4 = tup1 + tup2 + tup3
        train.append(tup4)  # ('G721', 'DO144', '0')
# 测试集        
test = []
for i in range(len(X_test)):
        tup1 = (X_test[i].split()[0].split(",")[0],)
        tup2 = (X_test[i].split()[0].split(",")[1],)
        tup3 = (y_test[i].split()[0],)
        tup4 = tup1 + tup2 + tup3
        test.append(tup4)  # ('G721', 'DO144', '0')

if __name__ == "__main__":
    G = get_graph_0()
    # G = get_graph()

    model = LINE(G, embedding_size=64, order='second')
    model.train(batch_size=1024, epochs=1, verbose=2)
    embeddings = model.get_embeddings()
    embeddings_data = np.array(embeddings)
    # 训练集
    X_train_gd = []
    for i in train:
        gd_feature = np.concatenate((embeddings[i[0]],embeddings[i[1]]))
        X_train_gd.append(gd_feature) 
    X_train_gd = np.array(X_train_gd)
    np.save('data/X_train_gmd_line.npy',X_train_gd)
    print(X_train_gd.shape)
    # 测试集
    X_test_gd = []
    for i in test:
        gd_feature = np.concatenate((embeddings[i[0]],embeddings[i[1]]))
        X_test_gd.append(gd_feature) 
    X_test_gd = np.array(X_test_gd)
    np.save('data/X_test_gmd_line.npy',X_test_gd)
    print(X_test_gd.shape)

    # with open("./data/gd_feature_line.txt","a") as f:
    #     print("开始写入文件！！！")
    #     for node,embed in embeddings.items():
    #         e = list(embed)
    #         e.insert(0, node)
    #         # lambda 冒号前是参数，可以有多个，用逗号隔开，冒号右边的为表达式。其实lambda返回值是一个函数的地址，也就是函数对象。
    #         e = list(map(lambda x:str(x), e))   # map() 会根据提供的函数对指定序列做映射。
    #         f.writelines(" ".join(e) + "\n")
    print("完成写入！！！")


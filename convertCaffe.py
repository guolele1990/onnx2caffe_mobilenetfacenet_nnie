from __future__ import print_function

import sys
import os
#caffe_root='/opt/caffe_plus/python'
caffe_root='/opt/caffe-1.0/python/'
# os.chdir(caffe_root)
sys.path.insert(0,caffe_root)
import caffe

from caffe.proto import caffe_pb2
import onnx

caffe.set_mode_cpu()
sys.path.append('../')
from onnx2caffe._transformers import ConvAddFuser, ConstantsToInitializers
from onnx2caffe._graph import Graph

import onnx2caffe._operators as cvt
import onnx2caffe._weightloader as wlr
from onnx2caffe._error_utils import ErrorHandling
from onnx import shape_inference
from modelComparator import compareOnnxAndCaffe

transformers = [
    ConstantsToInitializers(),
    ConvAddFuser(),
]


def convertToCaffe(graph, prototxt_save_path, caffe_model_save_path, exis_focus=False, focus_concat_name=None, focus_conv_name=None):  # 如果有 focus 层，自己添加参数
    exist_edges = []
    layers = []
    exist_nodes = []
    err = ErrorHandling()
    gap_kernel_shape = [1, 1]  # 定制化操作参数，不会通用, gap 的池化卷积层
    for i in graph.inputs:  # input 就是可视化中，第一个灰色东西，显示输入名 和 输入 shape，不是 op.
        edge_name = i[0]  # 一般是 images, data, input 这种名字

        input_layer = cvt.make_input(i)  # 生成 prototxt 风格的input

        layers.append(input_layer)
        exist_edges.append(i[0])
        graph.channel_dims[edge_name] = graph.shape_dict[edge_name][1]  # shape_dict[edge_name] 如 (1, 3, 112, 112) 这种

    for id, node in enumerate(graph.nodes):

        node_name = node.name  # node name 参数，就是节点在当前模型中的名字

        op_type = node.op_type  # op 类型，卷积， relu 这种

        if exis_focus:
            if op_type == "Slice":
                continue
            if node_name == focus_concat_name:
                converter_fn = cvt._ONNX_NODE_REGISTRY["PassThrough"]
                output_name = str(node.outputs[0])
                layer = converter_fn("focus", "images", output_name, 3, 2, 2)  # 3是输入通道，2 是 pytorch 中的步长
                #layers.append(layer)
                #exist_edges.append("47")
                if type(layer) == tuple:
                    for l in layer:  # 一般是 bn 层， caffe 中的 bn 是分为两部分， BN 和 Scale 层
                        #  print("layer.name = ", l.layer_name)
                        layers.append(l)
                else:
                    layers.append(layer)
                outs = node.outputs  # 节点输出名
                for out in outs:
                    exist_edges.append(out)
                continue
        if op_type == "Clip":  # relu6 在 onnx 里是 clip
            op_type = "Relu6"

        #print(node_name)
        inputs = node.inputs  # 列表，由可视化中 input 一栏中 name 字段组成，顺序同可视化界面一致。如果某个键有参数数组，则也会在 input_tensors 存在

        inputs_tensor = node.input_tensors  # 字典，可视化界面中，如果有参数数组就是这里面的值，键也在input 中， 有多少参数数组就有多少键值

        input_non_exist_flag = False

        for inp in inputs:  # input 组成元素有两种，一是上层节点 name，二是本层参数 name
            if inp not in exist_edges and inp not in inputs_tensor:  # 筛除，正常节点判断条件是不会成立的
                input_non_exist_flag = True
                break
        if input_non_exist_flag:
            continue

        if op_type not in cvt._ONNX_NODE_REGISTRY:  # 如果没在 op 字典中，报错
            err.unsupported_op(node)
            continue
        converter_fn = cvt._ONNX_NODE_REGISTRY[op_type]  # 相应转换函数
        if op_type == "GlobalAveragePool":
            layer = converter_fn(node, graph, err, gap_kernel_shape)
        else:
            #print("GlobalAveragePool  GlobalAveragePool")
            #print(op_type)
            layer = converter_fn(node, graph, err)
        if type(layer) == tuple:
            for l in layer:  # 一般是 bn 层， caffe 中的 bn 是分为两部分， BN 和 Scale 层
                #  print("layer.name = ", l.layer_name)
                layers.append(l)
        else:
            layers.append(layer)
        outs = node.outputs  # 节点输出名
        for out in outs:
            exist_edges.append(out)  # 储存输出节点，方便下面使用

    net = caffe_pb2.NetParameter()  # caffe 模型结构
    for id, layer in enumerate(layers):

        layers[id] = layer._to_proto()  # 转为 proto 风格？
        print(layers[id])
    net.layer.extend(layers)  # 将层名加入网络模型

    with open(prototxt_save_path, 'w') as f:  # 形成 prototxt 文件
        print(net, file=f)  # 写入 prototxt 文件
    # ------ 到此 prototxt 文件转换结束 ------
    # ------ 下面转换 caffemodel 文件 ------
    caffe.set_mode_cpu()
    deploy = prototxt_save_path
    net = caffe.Net(deploy,
                    caffe.TEST)

    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type

        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False
        if exis_focus:
            if op_type == "Slice":
                continue
        if op_type == "Clip":
            op_type = "Relu6"
        if op_type not in wlr._ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        #print(node_name)
        converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
        if node_name == focus_conv_name:
            converter_fn(net, node, graph, err, pass_through=1)
        else:
            converter_fn(net, node, graph, err)  # 复制模型参数

    net.save(caffe_model_save_path)  # 保存模型
    return net


def getGraph(onnx_path):
    model = onnx.load(onnx_path)
    #print(onnx.helper.printable_graph(model.graph))

    #model = shape_inference.infer_shapes(model)

    model_graph = model.graph
    graph = Graph.from_onnx(model_graph)
    graph = graph.transformed(transformers)
    graph.channel_dims = {}

    return graph


if __name__ == "__main__":

    onnx_source_dir = r'/opt/deeplearning/onnx2caffe_mobilenetfacenet_nnie/weights'
    save_dir = r"/opt/deeplearning/onnx2caffe_mobilenetfacenet_nnie/weights"
    onnx_name = 'S1.onnx'
    #onnx_source_dir = r'F:\demo\Pytorch_demo\yolov5-master\weights'
    #onnx_name = 'own_5s.onnx'
    onnx_path = os.path.join(onnx_source_dir, onnx_name)
    #onnx_path = "../models/onnx/version-RFB-320_simplified.onnx"
    prototxt_name = onnx_name.split('.')[0] + ".prototxt"
    caffemodel_name = onnx_name.split('.')[0] + ".caffemodel"
    prototxt_path = os.path.join(save_dir, prototxt_name)
    caffemodel_path = os.path.join(save_dir, caffemodel_name)
    graph = getGraph(onnx_path)

    convertToCaffe(graph, prototxt_path, caffemodel_path)
    print('Caffe model was saved to path : ' + onnx_source_dir)
    # compareOnnxAndCaffe(onnx_path, prototxt_path, caffemodel_path)



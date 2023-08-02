from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe import params as P
import math
import numpy as np
from ._graph import Node, Graph
from MyCaffe import Function as myf


def _compare(a, b, encoding="utf8"):  # type: (Text, Text, Text) -> bool
    if isinstance(a, bytes):
        a = a.decode(encoding)
    if isinstance(b, bytes):
        b = b.decode(encoding)
    return a == b


def make_input(input):
    name = input[0]
    output = input[0]
    output = [output]
    shape = input[2]
    shape = list(shape)
    input_layer = myf("Input", name, [], output, input_param=dict(shape=dict(dim=shape)))
    return input_layer


def _convert_conv(node, graph, err):
    weight_name = node.inputs[1]  # 0 是上层节点名，1 开始才是本层参数 name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))
    is_deconv = False
    if node.op_type.endswith("Transpose"):
        is_deconv = True
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    dilations = node.attrs.get("dilations", [1, 1])
    # groups = 1
    groups = node.attrs.get("group", 1)
    kernel_shape = node.attrs["kernel_shape"]
    pads = node.attrs.get("pads", [0, 0, 0, 0])
    strides = node.attrs["strides"]

    layer = myf("Convolution", node_name, [input_name], [output_name],
                kernel_h=kernel_shape[0], kernel_w=kernel_shape[1],
                stride_h=strides[0], stride_w=strides[1], group=groups,
                pad_h=pads[0], pad_w=pads[1],
                num_output=W.shape[0], dilation=dilations[0], bias_term=bias_flag)

    graph.channel_dims[output_name] = W.shape[0]
    return layer


def _convert_relu(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)

    if input_name == output_name:
        inplace = True
    else:
        inplace = False

    layer = myf("ReLU", name, [input_name], [output_name], in_place=inplace)
    # l_top_relu1 = L.ReLU(l_bottom, name=name, in_place=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer


def _convert_relu6(node, graph, err):
    relu6_input_name = str(node.inputs[0])
    relu6_output_name = str(node.outputs[0])
    old_name = str(node.name)
    name = old_name + "_relu6"
    layers = []

    # 首先做 relu，node 1
    relu_name = name + "_relu"

    layer = myf("ReLU", relu_name, [relu6_input_name], [relu_name + "_out"], in_place=False)
    graph.channel_dims[relu_name + "_out"] = graph.channel_dims[relu6_input_name]  # 输出节点的维度赋值
    layers.append(layer)

    # 其次做 threshold，node 2
    thre_name = name + "_thre"
    layer = myf("Threshold", thre_name, [relu_name + "_out"], [thre_name + "_out"],
                in_place=False,
                threshold_param=dict(threshold=6.0)  # 阈值，大于它输出 1，否则为 0
                )
    graph.channel_dims[thre_name + "_out"] = graph.channel_dims[relu_name + "_out"]
    layers.append(layer)

    # threshold 左输出做线性变化，与x相乘，node 3
    thre_left_power_name = name + "_thre_left_power"
    layer = myf("Power", thre_left_power_name, [thre_name + "_out"], [thre_left_power_name + "_out"],
                in_place=False,
                power_param=dict(
                    power=1.0,
                    scale=(-1.0),
                    shift=1.0
                )
                )
    graph.channel_dims[thre_left_power_name + "_out"] = graph.channel_dims[thre_name + "_out"]
    layers.append(layer)

    # relu 后 x 输出处理，node 4
    x_thre_out_name = name + "_x_mul_thre_out"
    layer = myf("Eltwise", x_thre_out_name, [relu_name + "_out", thre_left_power_name + "_out"],
                [x_thre_out_name + "_out"], operation=P.Eltwise.PROD)
    graph.channel_dims[x_thre_out_name + "_out"] = graph.channel_dims[relu_name + "_out"]
    layers.append(layer)

    # threshold 右输出做线性变化，node 5
    thre_right_power_name = name + "_thre_right_power"
    layer = myf("Power", thre_right_power_name, [thre_name + "_out"], [thre_right_power_name + "_out"],
                in_place=False,
                power_param=dict(
                    power=1.0,
                    scale=6.0,
                    shift=0.0
                )
                )
    graph.channel_dims[thre_right_power_name + "_out"] = graph.channel_dims[thre_name + "_out"]
    layers.append(layer)

    # 最后结果汇总，node 6
    add_name = name + "_add"
    layer = myf("Eltwise", add_name, [x_thre_out_name + "_out", thre_right_power_name + "_out"],
                [relu6_output_name], operation=P.Eltwise.SUM)
    graph.channel_dims[relu6_output_name] = graph.channel_dims[relu6_input_name]
    layers.append(layer)

    return tuple(layers)


def _convert_prelu(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)

    if input_name == output_name:
        inplace = True
    else:
        inplace = False

    layer = myf("PReLU", name, [input_name], [output_name], in_place=inplace)
    # l_top_relu1 = L.ReLU(l_bottom, name=name, in_place=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer

def _convert_leaky_relu(node,graph,err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)
    alpha = node.attrs.get("alpha", 1)

    print(node.attrs["alpha"])

    if input_name==output_name:
        inplace = True
    else:
        inplace = False

    layer = myf("ReLU",name,[input_name],[output_name],in_place=inplace, negative_slope=alpha)
    # l_top_relu1 = L.ReLU(l_bottom, name=name, in_place=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer
def _convert_sigmoid(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)

    if input_name == output_name:
        inplace = True
    else:
        inplace = False

    layer = myf("Sigmoid", name, [input_name], [output_name], in_place=inplace)
    # l_top_relu1 = L.ReLU(l_bottom, name=name, in_place=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer


def _convert_BatchNorm(node, graph, err):
    epsilon = node.attrs.get("epsilon", 1e-5)
    scale = node.input_tensors[node.inputs[1]]
    bias = node.input_tensors[node.inputs[2]]
    mean = node.input_tensors[node.inputs[3]]
    var = node.input_tensors[node.inputs[4]]
    node_name = node.name

    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])

    if input_name == output_name:
        inplace = True
    else:
        inplace = False

    bn_layer = myf("BatchNorm", node_name + "_bn", [input_name], [output_name], eps=epsilon, use_global_stats=True, in_place=inplace)
    scale_layer = myf("Scale", node_name, [output_name], [output_name], in_place=True, bias_term=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return bn_layer, scale_layer


def _convert_Add(node, graph, err):
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    node_name = node.name

    max_dim = 0
    for name in input_name_list:
        print(graph.channel_dims)
        #if graph.channel_dims[name] > max_dim:
           # max_dim = graph.channel_dims[name]

    if 'broadcast' in node.attrs:
        if node.attrs['broadcast'] == 1:
            input_node_number = len(input_name_list)
            if input_node_number != 2:
                return err.unsupported_op_configuration(node, "Broadcast Add must has 2 input, not {}".format(input_node_number))
            axis = node.attrs['axis']
            flat_layer = myf("Flatten", node_name + '_flat', [input_name_list[1]], [output_name + '_flat'])
            layer = myf("Bias", node_name, [input_name_list[0], output_name + '_flat'], [output_name], axis=axis)
            # layer = myf("Bias", node_name, input_name_list, [output_name], bias_term = False, axis = axis)
            graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]
            return flat_layer, layer

    layer = myf("Eltwise", node_name, input_name_list, [output_name], operation=P.Eltwise.SUM)
    graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]
    return layer


def _convert_Mul(node, graph, err):
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    node_name = node.name

    # max_dim = 0
    # for name in input_name_list:
    #     if graph.channel_dims[name]>max_dim:
    #         max_dim = graph.channel_dims[name]

    if 'broadcast' in node.attrs:
        if node.attrs['broadcast'] == 1:
            input_node_number = len(input_name_list)
            if input_node_number != 2:
                return err.unsupported_op_configuration(node, "Broadcast Mul must has 2 input, not {}".format(input_node_number))
            axis = node.attrs['axis']
            flat_layer = myf("Flatten", node_name + '_flat', [input_name_list[1]], [output_name + '_flat'])
            layer = myf("Scale", node_name, [input_name_list[0], output_name + '_flat'], [output_name], bias_term=False, axis=axis)
            graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]
            return flat_layer, layer

    layer = myf("Eltwise", node_name, input_name_list, [output_name], operation=P.Eltwise.PROD)
    graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]
    return layer



def _convert_Div_Add(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name

    input_1_shape = graph.shape_dict[node.inputs[1]]  # 获得 input_1 的 shape
    #print("node.input_tensors = ", node.input_tensors[node.inputs[1]])
    scale_ = 1.0
    shift_ = 0.0
    power_ = 1.0
    if node.op_type == "Div":
        scale_ = (1 / node.input_tensors[node.inputs[1]])
    elif node.op_type == "Add":

        #if len(input_1_shape) == 4:  # 如果是两个向量相加，比如 输入两组特征.  我也不记得当初为什么要这样写。。。。。修正了
        if len(node.input_tensors) == 0:
            layer = myf("Eltwise", node_name, node.inputs, [output_name], operation=P.Eltwise.SUM)
            # graph.channel_dims[output_name] = graph.channel_dims[input_name]
            graph.channel_dims[output_name] = input_1_shape
            return layer

        shift_ = node.input_tensors[node.inputs[1]]
    layer = myf("Power", node_name, [input_name], [output_name],
                in_place=False,
                power_param=dict(
                    power=power_,
                    scale=scale_,
                    shift=shift_
                )
                )
    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer


"slice layer : https://blog.csdn.net/qq_37118111/article/details/86609138"
def _convert_Split(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    print(node.attrs)
    print(node.outputs)
    layer = myf("Slice", node_name, [input_name], node.outputs,
                axis=node.attrs['axis'],
                slice_point=node.attrs['split'][0]
                )
    print(graph.channel_dims)
    graph.channel_dims[node.outputs[0]] = node.attrs['split'][0]
    graph.channel_dims[node.outputs[1]] = node.attrs['split'][1]
    print(graph.channel_dims)
    return layer


def _convert_Reshape(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    if len(node.inputs) == 1:
        shape = tuple(node.attrs.get('shape', ()))
    else:
        shape = tuple(node.input_tensors[node.inputs[1]])
    # if shape == ():

    if input_name == output_name:
        inplace = True
    else:
        inplace = False
    if len(shape) == 2:
        layer = myf("Flatten", node_name, [input_name], [output_name], in_place=inplace)
        graph.channel_dims[output_name] = shape[1]
        return layer
    elif len(shape) == 4 or len(shape) == 3 or len(shape) == 5:
        graph.channel_dims[output_name] = shape[1]
        layer = myf("Reshape", node_name, [input_name], [output_name], reshape_param=dict(shape=dict(dim=list(shape))))
        return layer
    else:
        return err.unsupported_op_configuration(node, "Reshape dimention number shall be 2 or 4")


def _convert_Flatten(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    # shape = tuple(node.attrs.get('shape', ()))
    if input_name == output_name:
        inplace = True
    else:
        inplace = False
    layer = myf("Flatten", node_name, [input_name], [output_name], in_place=inplace)
    # graph.channel_dims[output_name] = shape[1]
    return layer


def _convert_Permute(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    if len(node.inputs) == 1:
        shape = tuple(node.attrs.get('perm', ()))
    else:
        shape = tuple(node.input_tensors[node.inputs[1]])

    if len(shape) == 4 or len(shape) == 5:
        layer = myf("Permute", node_name, [input_name], [output_name], permute_param=dict(order=list(shape)))
        return layer
    else:
        return err.unsupported_op_configuration(node, "Reshape dimention number shall be 2 or 4")


def _convert_Softmax(node, graph, err):
    node_name = node.name
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    axis = node.attrs.get("axis", 1)

    layer = myf('Softmax', node_name, input_name_list, [output_name], axis=axis)

    return layer


def _convert_pool(node, graph, err, gap_kernel_shape=None):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    gap = False
    if node.op_type.endswith("MaxPool"):
        pool_type = P.Pooling.MAX

    elif node.op_type == "AveragePool":
        pool_type = P.Pooling.AVE

    elif node.op_type == "GlobalAveragePool":
        pool_type = P.Pooling.AVE
        gap = True

    else:
        return err.unsupported_op_configuration(node, "Unsupported pool type")

    if gap:
        kernel_shape = gap_kernel_shape  # gap_kernel_shape 是列表/元组
    else:
        kernel_shape = node.attrs["kernel_shape"]

    strides = node.attrs.get('strides', [1, 1])
    pads = node.attrs.get('pads', [0, 0, 0, 0])

    layer = myf("Pooling", node_name, [input_name], [output_name], pooling_param=dict(pool=pool_type,
                                                                                      kernel_h=kernel_shape[0],
                                                                                      kernel_w=kernel_shape[1],
                                                                                      stride_h=strides[0],
                                                                                      stride_w=strides[1],
                                                                                      pad_h=pads[0],
                                                                                      pad_w=pads[1]))
    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer


def _convert_dropout(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    ratio = node.attrs.get('ratio', 0.5)
    layer = myf("Dropout", node_name, [input_name], [output_name], dropout_ratio=ratio)
    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer


def _convert_gemm(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))
        return

    #if node.attrs["broadcast"] != 1 or node.attrs["transB"] != 1:
    if node.attrs["transB"] != 1:
        return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")

    b = None
    bias_flag = False
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]

    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    if b is not None:
        bias_flag = True
        if W.shape[0] != b.shape[0]:
            return err.unsupported_op_configuration(node,
                                                    "Gemm is supported only for inner_product layer")

    layer = myf("InnerProduct", node_name, [input_name], [output_name], num_output=W.shape[0], bias_term=bias_flag)
    graph.channel_dims[output_name] = W.shape[0]

    return layer


def _convert_matmul(node, graph, err):  # 建立网络结构图

    node_name = node.name
    input_name = str(node.inputs[0])  # 上层节点名
    output_name = str(node.outputs[0])  # 输出节点名
    weight_name = node.inputs[1]  # 本层参数名
    if weight_name in node.input_tensors:  # 判断参数数组是否真的存在
        W = node.input_tensors[weight_name]  # 获得参数数组
    else:  # 没有的话也就没意义继续了
        err.missing_initializer(node,
                                "MatMul weight tensor: {} not found in the graph initializer".format(weight_name, ))
        return

    b = None
    bias_flag = False
    if len(node.inputs) > 2:  # 如果只有上层节点名和 W 权值，则为 2
        b = node.input_tensors[node.inputs[2]]
    # 权值 shape 不对，也没意义继续了
    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(node, "MatMul is supported only for inner_product layer")
    if b is not None:
        bias_flag = True
        if W.shape[1] != b.shape[0]:  # FC 中，二者 shape[0] 是输出通道数， 一定相等，shape[1] 是输入通道数。
            return err.unsupported_op_configuration(node,
                                                    "MatMul is supported only for inner_product layer")
    # 不同于 gemm ，matmul 不做转置操作，w = (A, B), A 是输入通道数， B 是输出通道数
    layer = myf("InnerProduct", node_name, [input_name], [output_name], num_output=W.shape[1], bias_term=bias_flag)
    graph.channel_dims[output_name] = W.shape[1]  # 获得输出通道数

    return layer


def _convert_upsample(node, graph, err):

    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    # input_shape = graph.shape_dict[input_name]
    # channels = input_shape[1]
    channels = graph.channel_dims[input_name]

    # layer = myf("Deconvolution", node_name, [input_name], [output_name],
    #             kernel_size=2 * factor - factor % 2,
    #             stride=factor, group=channels,
    #             pad = pad, num_output=channels, bias_term = False)
    mode = node.attrs["mode"]
    print(node.name, "mode = ", mode)
    # https://github.com/pytorch/pytorch/issues/6900
    if mode == "bilinear":
        """
        factor = int(node.attrs["height_scale"])
        pad = int(math.ceil((factor - 1) / 2.))
        layer = myf("Deconvolution", node_name, [input_name], [output_name],
                    convolution_param=dict(
                        num_output=channels,
                        kernel_size=2 * factor - factor % 2,
                        stride=factor,
                        pad=pad,
                        group=channels,
                        bias_term=False,
                        weight_filler=dict(type="bilinear_upsampling")
                    ))
        """
    elif str(mode, encoding="gbk") == "nearest" or str(mode, encoding="gbk") == "linear":
        print(node.inputs[-1])
        scales = node.input_tensors.get(node.inputs[-1])
        print(scales)
        height_scale = scales[2]
        width_scale = scales[3]
        layer = myf("Upsample", node_name, [input_name], [output_name],
                    upsample_param=dict(
                        #mode=int(0),
                        #mode="nearest",
                        #height_scale=int(height_scale),
                        #width_scale=int(width_scale)
                        scale=2
                    ))
    else:
        factor = int(node.attrs["height_scale"])
        layer = myf("Deconvolution", node_name, [input_name], [output_name],
                    convolution_param=dict(
                        num_output=channels,
                        kernel_size=factor,
                        stride=factor,
                        group=channels,
                        bias_term=False,
                    ))

    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer


def _convert_concat(node, graph, err):
    node_name = node.name
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    axis = node.attrs.get("axis", 1)

    layer = myf('Concat', node_name, input_name_list, [output_name], axis=axis)
    if axis == 1:
        dim = 0
        for name in input_name_list:
            print(name+":"+" ",graph.channel_dims[name])
            if isinstance(graph.channel_dims[name], tuple):
                print(name+":"+" is a tuple")
                dim += graph.channel_dims[name][axis]
            else:
                dim += graph.channel_dims[name]
        graph.channel_dims[output_name] = dim
    else:
        graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]

    return layer


def _convert_conv_transpose(node, graph, err):  # 反卷积
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    weight_name = node.inputs[1]
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    dilations = node.attrs.get("dilations", [1, 1])
    # groups = 1
    groups = node.attrs.get("group", 1)
    kernel_shape = node.attrs["kernel_shape"]
    pads = node.attrs.get("pads", [0, 0, 0, 0])
    strides = node.attrs["strides"]

    layer = myf('Deconvolution', node_name, [input_name], [output_name],
                convolution_param=dict(
                    num_output=W.shape[0],  # 此处我做了更改，分组卷积需要注意
                    kernel_h=kernel_shape[0], kernel_w=kernel_shape[1],
                    stride_h=strides[0], stride_w=strides[1],
                    group=groups,
                    pad_h=pads[0], pad_w=pads[1],
                    bias_term=bias_flag,
                ))

    graph.channel_dims[output_name] = W.shape[1]
    return layer


def _convert_PassThrough(node_name, input_name, output_name, input_channel, block_height, block_width):  # 反卷积

    layer = myf('PassThrough', node_name, [input_name], [output_name],
                pass_through_param=dict(
                    num_output=input_channel * block_height * block_width,
                    block_height=block_height,
                    block_width=block_width,
    ))

    return layer
def _convert_conv_slice(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    axes = node.attrs.get('axes', [])
    #graph.channel_dims[input_name] = graph.shape_dict[input_name][1]
    #channels = graph.shape_dict[input_name][1]
    #channels = graph.channel_dims[input_name]
    channels = graph.shape_dict[input_name][1]

    if len(axes) != 1:
        return err.unsupported_op_configuration(node, "Only single axis Slice is supported now")
    starts = node.attrs['starts']
    ends = node.attrs['ends']
    axes = node.attrs.get('axes', [])
    start = starts[0]
    end = ends[0]
    valid_pts = []
    for pt in [start, end]:
        if pt is not None and pt != 0 and pt != channels:
            valid_pts.append(pt)
    if start == 0:
        output_name_list = [output_name, str(output_name) + "slice_another"]
    else:
        output_name_list = [str(output_name) + "slice_another", output_name]
    if len(axes) == 0: axes = range(len(starts))
    if len(axes) == 1:
        if axes[0] == 0:
            axis = 'batch'
        elif axes[0] == 1:
            axis = 'channel'
        elif axes[0] == 2:
            axis = 'height'
        elif axes[0] == 3:
            axis = 'width'
        else:
            return err.unsupported_op_configuration(node, "Slice is supported only along H, W or C dimensions")
    else:
        return err.unsupported_op_configuration(node,"Slice is supported only along one axis for 3D or 4D Tensors")
    layer = myf('Slice', node_name, [input_name], output_name_list, slice_dim=axes[0], slice_point=valid_pts)
    graph.channel_dims[output_name_list[0]] = valid_pts[0]
    graph.channel_dims[output_name_list[-1]] = channels - valid_pts[-1]
    return layer


_ONNX_NODE_REGISTRY = {
    "Conv": _convert_conv,
    "Relu": _convert_relu,
    "Relu6": _convert_relu6,
    "PRelu": _convert_prelu,
    "LeakyRelu": _convert_leaky_relu,
    "BatchNormalization": _convert_BatchNorm,
    "Add": _convert_Div_Add,
    "Mul": _convert_Mul,
    "Div": _convert_Div_Add,
    "Reshape": _convert_Reshape,
    "MaxPool": _convert_pool,
    "AveragePool": _convert_pool,
    "GlobalAveragePool": _convert_pool,
    "Dropout": _convert_dropout,
    "Gemm": _convert_gemm,
    "MatMul": _convert_matmul,
    "Upsample": _convert_upsample,
    "Resize":_convert_upsample,
    "Concat": _convert_concat,
    "ConvTranspose": _convert_conv_transpose,
    "Sigmoid": _convert_sigmoid,
    "Flatten": _convert_Flatten,
    "Transpose": _convert_Permute,
    "Softmax": _convert_Softmax,
    "Split": _convert_Split,
    "PassThrough": _convert_PassThrough,
    "Slice": _convert_conv_slice,
}

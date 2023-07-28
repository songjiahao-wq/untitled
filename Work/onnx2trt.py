import onnx
import tensorrt as trt
# import sys
# sys.setrecursionlimit(500000)
import logging

def onnx_export_engine(model_name, workspace, half=False):
    # path='weights/cam_fusion_net.onnx'
    # filename='onnxmodel'
    # model_name='breath_cls'
    # model_name='facedect'

    path='test2/'+model_name+'.onnx'
    path = r'D:\project\T2M-GPT2\weights\T2M-GPT-vqvae.onnx'
    #创建构建器
    logger=trt.Logger(trt.Logger.ERROR)
    builder=trt.Builder(logger)
    #创建一个构建配置
    config=builder.create_builder_config()
    # config.max_workspace_size=workspace*1<<30
    #创建网络定义
    flag=(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network=builder.create_network(flag)
    #导入onnx模型
    parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(path)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs=[network.get_input(i) for i in range(network.num_inputs)]
    # print('inputs',inputs)
    outputs=[network.get_output(i) for i in  range(network.num_outputs)]
    # network.get_input(0).setAllowedFormats(int)
    # network.get_input(1).setAllowedFormats(int)
    profile = builder.create_optimization_profile()
    # profile.set_shape('idx', (1, 1), (1, 20), (1, 55))# trans
    profile.set_shape('input', (1, 1), (1, 20), (1, 55))# resnet
    # profile.set_shape("index", (1, ), (1, ), (1, ))
    config.add_optimization_profile(profile)
    # for inp in inputs:
    #     LOGGER.info(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
    # for out in outputs:
    #     LOGGER.info(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')
    #
    # LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 else 32} engine in {f}')
    logging.info(f'Tensorrt building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    engine_path=model_name+'.engine'
    # with builder.build_engine(network,config) as engine:
    with builder.build_serialized_network(network,config) as engine:
        with open(engine_path,'wb') as t:
            # t.write(engine.serialize())
            t.write(engine)
    print('转化完成')

if __name__ == '__main__':
    # model_names = ['modified_stable_diffusion']
    model_names = ['resnet']
    for modelname in model_names:
        onnx_export_engine(modelname, 4, half=False)
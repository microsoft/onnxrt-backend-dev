from onnx import ModelProto
from onnx.inline import inline_local_functions


def inline_model_proto(model_proto: ModelProto) -> ModelProto:
    """
    Inlines a model.

    :param model_proto: ModelProto
    :return: inlined model
    """
    # model = onnx.load(input_file_name, load_external_data=False)
    return inline_local_functions(model_proto)


def optimize_model_proto(model_proto: ModelProto) -> ModelProto:
    """
    Optimizes a model proto to optimize onnxruntime.

    :param model_proto: ModelProto
    :return: optimized model

    You should run that before calling this function

    ::

        model_proto = exported.to_model_proto(
            opset_version=self._resolved_onnx_exporter_options.onnx_registry.opset_version
        )
    """
    from onnxrewriter.optimizer import optimize
    from onnxrewriter.rewriter.transformers import rewrite

    model_proto = inline_model_proto(model_proto)
    model_proto = optimize(
        model_proto,
        num_iterations=2,
        onnx_shape_inference=False,
        function_aware_folding=True,
    )
    model_proto = rewrite(model_proto)

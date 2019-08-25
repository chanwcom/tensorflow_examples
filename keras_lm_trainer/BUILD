package(
    default_visibility = ["//visibility:public"]
)

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")


py_binary(
    name = "lm_trainer",
    srcs = [ 
        "lm_trainer.py",
    ],  
    deps = [ 
        ":lm_model",
        "//speech/trainer/data_queue:word_id_example_queue",
        "//speech/trainer/data_queue:word_id_example_queue_config_py_proto",
    ],  
)

py_test(
    name = "keras_lm_trainer_test",
    srcs = [
        "keras_lm_trainer_test.py"
    ],
    deps = [ 
        ":keras_lm_trainer_config_py_proto",
        ":keras_lm_trainer",
    ],  
)

py_library(
    name = "lm_model",
    srcs = [
        "lm_model.py",
    ],
    deps = [
    ],
)

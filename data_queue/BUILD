package(
    default_visibility = ["//visibility:public"]
)

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

proto_library(
    name = "example_queue_config_proto",
    srcs = [ 
        "example_queue_config.proto"
    ],  
)

cc_proto_library(
    name = "example_queue_config_cc_proto",
    deps = [ 
        ":example_queue_config_proto",
    ],  
)

py_proto_library(
    name = "example_queue_config_py_proto",
    srcs = [ 
        "example_queue_config.proto"
    ],  
)

py_library(
    name = "example_queue",
    srcs = [
        "example_queue.py"
    ],
    deps = [
        ":example_queue_config_py_proto",
        "//speech/common:utterance_data_proto_helper",
        "//speech/common:utterance_data_py_pb2",
        "//speech/feature:extract_feature",
    ],
)

py_test(
    name = "example_queue_test",
    srcs = [
        "example_queue_test.py"
    ],
    data = [
        "testdata",
    ],
    deps = [ 
        ":create_test_tfrecord",
        ":example_queue_config_py_proto",
        ":example_queue",
        "//signal_processing:array_test_py",
        "//speech/common:utterance_data_proto_helper",
        "//speech/feature:extract_feature",
    ],  
)

py_binary(
    name = "create_test_tfrecord",
    srcs = [ 
        "create_test_tfrecord.py"
    ],  
    deps = [ 
        "//speech/common:utterance_data_py_pb2",
        "@com_google_protobuf//:protobuf_python",
    ],  
)

proto_library(
    name = "word_id_example_queue_config_proto",
    srcs = [ 
        "word_id_example_queue_config.proto"
    ],  
)

cc_proto_library(
    name = "word_id_example_queue_config_cc_proto",
    deps = [ 
        ":word_id_example_queue_config_proto",
    ],  
)

py_proto_library(
    name = "word_id_example_queue_config_py_proto",
    srcs = [ 
        "word_id_example_queue_config.proto"
    ],  
)

py_library(
    name = "word_id_example_queue",
    srcs = [
        "word_id_example_queue.py",
    ],
    deps = [
        ":example_queue",
        ":word_id_example_queue_config_py_proto",
    ],
)

py_test(
    name = "word_id_example_queue_test",
    srcs = [
        "word_id_example_queue_test.py"
    ],
    deps = [ 
        ":word_id_example_queue_config_py_proto",
        ":word_id_example_queue",
    ],  
)

py_library(
    name = "word_id_example_queue_v2",
    srcs = [
        "word_id_example_queue_v2.py",
    ],
    deps = [
        ":example_queue",
        ":word_id_example_queue_config_py_proto",
    ],
)

py_test(
    name = "word_id_example_queue_v2_test",
    srcs = [
        "word_id_example_queue_v2_test.py"
    ],
    deps = [ 
        ":word_id_example_queue_config_py_proto",
        ":word_id_example_queue_v2",
    ],  
)

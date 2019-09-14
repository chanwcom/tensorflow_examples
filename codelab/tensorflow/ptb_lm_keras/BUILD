package(
    default_visibility = ["//visibility:public"]
)

py_library(
    name = "lm_model",
    srcs = [
        "lm_model.py",
    ],
)

py_library(
    name = "trainer",
    srcs = [
        "trainer.py",
    ],
    deps = [
        ":lm_model",
        "//speech/trainer/data_queue:example_queue_config_py_proto",
        "//speech/trainer/data_queue:word_id_example_queue_v2",
    ],
)

#py_test(
#    name = "example_queue_test",
#    srcs = [
#        "example_queue_test.py"
#    ],
#    data = [
#        "testdata",
#    ],
#    deps = [ 
#        ":create_test_tfrecord",
#        ":example_queue_config_py_proto",
#        ":example_queue",
#        "//signal_processing:array_test_py",
#        "//speech/common:utterance_data_proto_helper",
#        "//speech/feature:extract_feature",
#    ],  
#)

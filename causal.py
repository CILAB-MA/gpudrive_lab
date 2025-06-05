import tensorflow as tf
from causal_labels_pb2 import CausalLabels
import os
from waymo_open_dataset.protos import scenario_pb2
def parse_tfrecord(example_proto):
    label = CausalLabels()
    label.ParseFromString(example_proto.numpy())

    scenario_id = label.scenario_id
    results = label.labeler_results  # labeler_results는 이미 LabelerResults의 리스트입니다.

    print(f'Scenario ID: {scenario_id}')
    # print('Results', results)
    for idx, result in enumerate(results):
        print(f'Result {len(results)}: {result.causal_agent_ids}')

def as_proto_iterator(tf_dataset):
    """Parse the tfrecord dataset into a protobuf format."""
    for tfrecord in tf_dataset:
        # Parse the scenario protobuf
        scene_proto = scenario_pb2.Scenario()
        scene_proto.ParseFromString(bytes(tfrecord.numpy()))
        yield scene_proto

filename = '/data/casual/causal_labels.tfrecord'
dataset = tf.data.TFRecordDataset(filename)

for raw_example in dataset.take(2):
    parse_tfrecord(raw_example)

scenename = '/data/casual/v1.1/validation/RemoveCausal'
scene_dir = os.listdir(scenename)

tfrecord_dataset = tf.data.TFRecordDataset(os.path.join(scenename,scene_dir[0]), compression_type="")
tf_dataset_iter = as_proto_iterator(tfrecord_dataset)
for scene_proto in tf_dataset_iter:
    print(scene_proto)
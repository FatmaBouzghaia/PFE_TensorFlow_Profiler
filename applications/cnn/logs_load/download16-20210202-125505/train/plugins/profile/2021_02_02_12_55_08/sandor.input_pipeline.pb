$	*Vx�#@P���O�?6\��@!�^�@$	O48@��g��@���ڍ@!	�X�7@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails63��30@�[Ɏ�@�?1g�CV�?A'��@j�?I����=@Y{�%��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8$�@�x@\!��V�?1�T���N�?A�X��;�?I�'�$��?Y(��&2s�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8$D��R@��e6��?15���:U�?A�.��?I(�4�\@Y��,z��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8r�j��@���U���?1�'�.�`�?A�0���?I�wF[��@Y즔�J��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�^�@�!����?1�|a2U�?A<K�P��?I_~�Ɍ7@Y�vKr���?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�W��:@n��E���?1���o_�?A��w�?IA�C�� @Y �]����?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8��Q�b@n��׫�?1�q75P�?A��^��W�?I�C?���?Y�K�;���?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8y�'er@��KU�b�?1����^�?A8I�Ǵ6�?I����� @Y��y7�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�i����@M�x$^�?1�����W�?A;���R��?I���� @Y*8� "�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8	���h@��,'�t�?1e���m�?A̛õ���?I[���i@Y�F����?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8
A�]��@��v����?1�!��u`�?A������?I�gsb@Ye��2�P�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8z��C5e@o�$���?1	m9�b�?Ac'���?I�ݮ���?Y���߆�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�%�x@@�ի��?1���מY�?A�|^���?IAd�&ށ�?Yn��)��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�m�8)@�u?T��?1�q75P�?A���߽��?I     �@Y�!�
�l�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8��q��9@����5"�?1l\��O�?A�zܷZ'�?I�(�'r@Y2 Tq��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�;2V�@��ܴ��?1��H�H�?A�q�	�O�?I����/�@Ys��A�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8���r�G@�jI��?1�k���P�?Au�i���?Il��C�@YPō[���?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8'"��@��Coq�?1�׼��Z�?Aũ��,��?I��E�nT@Y�k
dv�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails80� ��w@�в��?1�Z��8�?Aw/��Q��?I� ��*:@Y�SW>��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8��"�tU@(F�̱��?1�N]�,O�?A��w��?I�o
+�@Yof����?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails86\��@��2���?1u ��W�?AD�H����?I��I�?Y+���+��?*	���x��@2F
Iterator::Model�]/M��?!>�]=��I@)�s�f���?1�Ji��:@:Preprocessing2U
Iterator::Model::ParallelMapV2�G�3�9�?!��Q��8@)�G�3�9�?1��Q��8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��q�_�?!4��$QZ:@)��I}Y��?1��%Ʃ5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���O��?!�t��kSH@)���'+��?1�貅p%@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap>�x��?!���('@)��}�u�?1ə�}� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*��g����?!�?,�@)��g����?1�?,�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�-�R\U�?!�pC;�`@)�-�R\U�?1�pC;�`@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�53.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t18.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9������@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	r�!��?F���7�?�[Ɏ�@�?!�!����?	!       "$	���LV�?k}p��V?�Z��8�?!e���m�?*	!       2$	�v ���?��A}���?'��@j�?!u�i���?:$	����h@�)n|a�?�ݮ���?!l��C�@B	!       J$	׈�]e�?i%��&�?+���+��?!{�%��?R	!       Z$	׈�]e�?i%��&�?+���+��?!{�%��?JGPUY������@b 
import tensorflow as tf


class NestedTensorsAccumulator:
    """
    Allows appending of tensors in a nested structure to be appended to TensorArrays structured in the same way
    """

    def __init__(self, array_size=None):
        self._array_size = array_size

        self._nested_tensor_arrays = None
        self._index = -1

    def zip_in(self, nested_tensors):
        """

        Add tensors in the given nested structure to be appended to TensorArrays structured in the same way

        :param nested_tensors:

        :return:
        """
        if not isinstance(nested_tensors, (tuple, list, dict)):
            raise ValueError(
                f"Main type of nested_tensors must be "
                f"tuple, list or dict, not {type(nested_tensors)}"
            )

        self._index += 1
        if self._array_size is not None and self._index >= self._array_size:
            raise Exception(
                f"Unable to append new nested tensors item, "
                f"maximum size of {self._array_size} reached."
            )

        self._nested_tensor_arrays = self._nest_to_zip_in(
            nested_tensors,
            current_nested_tensor_arrays=self._nested_tensor_arrays,
            new_nested_tensor_arrays=nested_tensors.__class__()
        )

    def unzip(self):

        return [self._rebuild_nested_tensors(
            idx,
            self._nested_tensor_arrays,
            self._nested_tensor_arrays.__class__(),
        ) for idx in range(self._index+1)]

    def _nest_to_zip_in(self, nested_tensors, current_nested_tensor_arrays, new_nested_tensor_arrays):
        if self._index > 0:
            self._check_types(nested_tensors, current_nested_tensor_arrays)

        if isinstance(nested_tensors, tuple):
            nest_to_zip_in_func = self._nest_to_zip_in_tuple
        elif isinstance(nested_tensors, list):
            nest_to_zip_in_func = self._nest_to_zip_in_list
        elif isinstance(nested_tensors, dict):
            nest_to_zip_in_func = self._nest_to_zip_in_dict
        else:
            raise Exception(f"Unexpected: nested_tensors ({type(nested_tensors)}) must be a tuple, list or dict.")

        return nest_to_zip_in_func(nested_tensors, current_nested_tensor_arrays, new_nested_tensor_arrays)

    def _nest_to_zip_in_tuple(self, nested_tensors, current_nested_tensor_arrays, new_nested_tensor_arrays):
        for v_idx, v in nested_tensors:

            def get_value():
                return current_nested_tensor_arrays[v_idx]

            nested_location_description = f"Tuple[{v_idx}]"

            v_array_current = self._get_current_nested_tensor_arrays_value(
                get_value,
                nested_location_description)

            v_array_new = self._create_new_nested_tensor_arrays_value(
                v,
                v_array_current,
                nested_location_description
            )

            new_nested_tensor_arrays += (v_array_new,)

        return new_nested_tensor_arrays

    def _nest_to_zip_in_list(self, nested_tensors, current_nested_tensor_arrays, new_nested_tensor_arrays):
        for v_idx, v in nested_tensors:

            def get_value():
                return current_nested_tensor_arrays[v_idx]

            nested_location_description = f"Tuple[{v_idx}]"

            v_array_current = self._get_current_nested_tensor_arrays_value(
                get_value,
                nested_location_description)

            v_array_new = self._create_new_nested_tensor_arrays_value(
                v,
                v_array_current,
                nested_location_description
            )

            new_nested_tensor_arrays.append(v_array_new)

        return new_nested_tensor_arrays

    def _nest_to_zip_in_dict(self, nested_tensors, current_nested_tensor_arrays, new_nested_tensor_arrays):
        for k, v in nested_tensors.items():
            def get_value():
                return current_nested_tensor_arrays[k]

            nested_location_description = f"Dict['{k}']"

            v_array_current = self._get_current_nested_tensor_arrays_value(
                get_value,
                nested_location_description)

            v_array_new = self._create_new_nested_tensor_arrays_value(
                v,
                v_array_current,
                nested_location_description
            )

            new_nested_tensor_arrays[k] = v_array_new

        return new_nested_tensor_arrays

    def _check_types(self, nested_tensors, nested_tensor_arrays):
        if not (type(nested_tensors) is type(nested_tensor_arrays)):
            raise ValueError(
                f"Structure of nested_tensors changed, unable to zip in nested tensors. "
                f"At current nesting level type of nested_tensors ({type(nested_tensors)}) is "
                f"not equal to type of nested_tensor_arrays ({type(nested_tensor_arrays)})."
            )

    def _get_current_nested_tensor_arrays_value(
            self,
            get_func,
            nested_location_description="[UNKNOWN]"
    ):

        nld = nested_location_description
        if self._index == 0:
            # replicating the nested_tensors structure for the first time
            return None
        else:
            try:
                return get_func()
            except Exception as e:
                raise ValueError(
                    f"Structure of nested_tensors changed, unable to zip in nested tensors. "
                    f"At current nesting level, tried to get current nested_tensor_arrays at {nld} but failed."
                ) from e

    def _create_tensor_array_for(self, value):
        if isinstance(value, (float, int)):
            dtype = tf.float32 if type(value) is float else tf.int64
        elif isinstance(value, tf.Tensor):
            dtype = value.dtype
        else:
            raise ValueError(f"Values in nested structure can only be float, int or tf.Tensor, "
                             f"value type is : {type(value)}")

        return tf.TensorArray(dtype, size=self._array_size)

    def _create_new_nested_tensor_arrays_value(
            self,
            nested_tensor_value,
            nested_array_value_current,
            nested_location_description='[UNKNOWN]'
    ):

        v = nested_tensor_value
        avc = nested_array_value_current
        nld = nested_location_description

        if isinstance(v, (tuple, list, dict)):
            return self._nest_to_zip_in(v, avc, v.__class__())
        else:
            tensor_array = None
            if self._index == 0:
                tensor_array = self._create_tensor_array_for(v)
            elif isinstance(avc, tf.TensorArray) and isinstance(v, tf.Tensor):
                tensor_array = avc

            if tensor_array is not None:
                return tensor_array.write(self._index, v)
            else:
                raise ValueError(
                    f"Structure of nested_tensors changed, unable to zip in nested tensors. "
                    f"At current nesting level, at {nld} the nested_tensors value ({type(v)}) "
                    f"is expected to be a tf.Tensor. "
                    f"At current nesting level, at {nld} the nested_tensor_arrays value ({type(avc)}) "
                    f"is expected to be a TensorArray."
                )

    def _rebuild_nested_tensors(self, index, nested_tensor_arrays, nested_tensors):
        if isinstance(nested_tensor_arrays, tuple):
            rebuild_nested_tensors_func = self._rebuild_nested_tensors_in_tuple
        elif isinstance(nested_tensor_arrays, list):
            rebuild_nested_tensors_func = self._rebuild_nested_tensors_in_list
        elif isinstance(nested_tensor_arrays, dict):
            rebuild_nested_tensors_func = self._rebuild_nested_tensors_in_dict
        else:
            raise Exception(f"Unexpected: nested_tensor_arrays has an unexpected type: "
                            f"({type(self._nested_tensor_arrays)}).")

        return rebuild_nested_tensors_func(index, nested_tensor_arrays, nested_tensors)

    def _rebuild_nested_tensors_in_tuple(self, index, nested_tensor_arrays, nested_tensor):
        for v in nested_tensor_arrays:
            if isinstance(v, tf.TensorArray):
                nested_tensor += (v.read(index),)
            else:
                nested_tensor += (self._rebuild_nested_tensors(index, v, v.__class__()),)

        return nested_tensor

    def _rebuild_nested_tensors_in_list(self, index, nested_tensor_arrays, nested_tensor):
        for v in nested_tensor_arrays:
            if isinstance(v, tf.TensorArray):
                nested_tensor.append(v.read(index))
            else:
                nested_tensor.append(self._rebuild_nested_tensors(index, v, v.__class__()))

        return nested_tensor

    def _rebuild_nested_tensors_in_dict(self, index, nested_tensor_arrays, nested_tensor):
        for k, v in nested_tensor_arrays.items():
            if isinstance(v, tf.TensorArray):
                nested_tensor[k] = v.read(index)
            else:
                nested_tensor[k] = self._rebuild_nested_tensors(index, v, v.__class__())

        return nested_tensor




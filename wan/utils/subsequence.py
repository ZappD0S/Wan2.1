import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _find_subsequence_index(main_array, sub_array):
    """
    (Helper Function) Finds the first index of a subarray using sliding_window_view.
    Returns -1 if not found.
    """
    window_shape = len(sub_array)
    # Handle edge cases
    if window_shape == 0:
        return 0  # An empty subsequence is found at the beginning
    if window_shape > len(main_array):
        return -1  # Subsequence cannot be longer than the array

    # Create a view of all subarrays (windows)
    windows = sliding_window_view(main_array, window_shape=window_shape)

    # Perform a vectorized comparison to find the match
    matches = np.all(windows == sub_array, axis=1)
    indices = np.where(matches)[0]

    # Return the first index found, or -1
    return indices[0] if len(indices) > 0 else -1


def get_nested_subsequence_mask(main_sequence, nested_subsequences):
    """
    Finds a sequence of nested subsequences and returns a boolean NumPy mask
    marking the position of the final subsequence within the original sequence.

    This function accepts either Python lists or NumPy arrays as input.
    """
    # 1. Ensure inputs are NumPy arrays for efficient processing.
    main_array = np.asarray(main_sequence)

    if not nested_subsequences:
        raise ValueError("The list of subsequences is empty!")

    subsequences_arrays = [np.asarray(sub) for sub in nested_subsequences]

    # 2. Core search logic
    current_array_view = main_array
    overall_offset = 0

    for sub_array in subsequences_arrays:
        # Use the helper to find the index within the current view
        index = _find_subsequence_index(current_array_view, sub_array)

        if index == -1:
            raise ValueError("Subsequence not found!")

        # Update the total offset from the start of the original array
        overall_offset += index
        # Narrow the search area to the subsequence just found
        current_array_view = current_array_view[index : index + len(sub_array)]

    # 3. If all subsequences were found, create the final mask
    final_mask = np.zeros_like(main_array, dtype=bool)
    final_subsequence_len = len(subsequences_arrays[-1])
    final_mask[overall_offset : overall_offset + final_subsequence_len] = True

    return final_mask

def mergeSort(arr):
    if len(arr) <= 1:
        return arr
    mid = int(len(arr) / 2)
    left = mergeSort(arr[:mid])
    right = mergeSort(arr[mid:])
    result = merge(left, right)
    return result


def merge(arr1, arr2):
    temp_result = []
    i = 0
    j = 0
    while i < len(arr1) or j < len(arr2):
        if i >= len(arr1):
            temp_result.append(arr2[j])
            j += 1
        elif j >= len(arr2):
            temp_result.append(arr1[i])
            i += 1
        elif arr1[i] <= arr2[j]:
            temp_result.append(arr1[i])
            i += 1
        else:
            temp_result.append(arr2[j])
            j += 1
    return temp_result


if __name__ == "__main__":
    input_arr = [5,2,1,3,5,6,7]
    print(mergeSort(input_arr))
#pragma once


namespace cpu_sort_lib
{
    template<typename T>
    void insertionSort(T arr[], int ind[], int n)
    {
        int i, j;
        int indKey;
        T key;
        for (i = 1; i < n; i++) {
            key = arr[i];
            indKey = ind[i];
            j = i - 1;

            /* Move elements of arr[0..i-1], that are
            greater than key, to one position ahead
            of their current position */
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                ind[j + 1] = ind[j];
                j = j - 1;
            }
            arr[j + 1] = key;
            ind[j + 1] = indKey;
        }
    }


    template<typename T>
     void insertionSort(T arr[], int n)
    {
        // printf("Sorting\n   %d ", n);
        int i, j;
        T key;
        for (i = 1; i < n; i++) {
            key = arr[i];
            j = i - 1;
            /* Move elements of arr[0..i-1], that are
            greater than key, to one position ahead
            of their current position */
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
    }


    template<typename T>
     void swap(T* a, T* b)
    {
        T t = *a;
        *a = *b;
        *b = t;
    }

    /* This function takes last element as pivot, places
    the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
    to left of pivot and all greater elements to right
    of pivot */
    template<typename T>
     int partition(T arr[], int low, int high)
    {
        T pivot = arr[high]; // pivot 
        int i = (low - 1); // Index of smaller element and indicates the right position of pivot found so far

        for (int j = low; j <= high - 1; j++)
        {
            // If current element is smaller than the pivot 
            if (arr[j] < pivot)
            {
                i++; // increment index of smaller element 
                swap(&arr[i], &arr[j]);
            }
        }
        swap(&arr[i + 1], &arr[high]);
        return (i + 1);
    }

    /* The main function that implements QuickSort
    arr[] --> Array to be sorted,
    low --> Starting index,
    high --> Ending index */
    template<typename T>
     void quickSort(T arr[], int low, int high)
    {
        if (low < high)
        {
            /* pi is partitioning index, arr[p] is now
            at right place */
            int pi = partition(arr, low, high);

            // Separately sort elements before 
            // partition and after partition 
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    template<class T>
     void quickSort(T arr[], int size)
    {
        quickSort(arr, 0, size - 1);
    }


    template<class T>
     void shellSort(T arr[], int n)
    {
        // Start with a big gap, then reduce the gap
        for (int gap = n / 2; gap > 0; gap /= 2)
        {
            // Do a gapped insertion sort for this gap size.
            // The first gap elements a[0..gap-1] are already in gapped order
            // keep adding one more element until the entire array is
            // gap sorted 
            for (int i = gap; i < n; i += 1)
            {
                // add a[i] to the elements that have been gap sorted
                // save a[i] in temp and make a hole at position i
                T temp = arr[i];

                // shift earlier gap-sorted elements up until the correct 
                // location for a[i] is found
                int j;
                for (j = i; j >= gap && arr[j - gap] > temp; j -= gap)
                    arr[j] = arr[j - gap];

                //  put temp (the original a[i]) in its correct location
                arr[j] = temp;
            }
        }
    }






    template<class T>

     void heapify(T arr[], int N, int i)
    {
        // Find largest among root, left child and right child

        // Initialize largest as root
        int largest = i;

        // left = 2*i + 1
        int left = 2 * i + 1;

        // right = 2*i + 2
        int right = 2 * i + 2;

        // If left child is larger than root
        if (left < N && arr[left] > arr[largest])

            largest = left;

        // If right child is larger than largest
        // so far
        if (right < N && arr[right] > arr[largest])

            largest = right;

        // Swap and continue heapifying if root is not largest
        // If largest is not root
        if (largest != i) {

            swap(&arr[i], &arr[largest]);

            // Recursively heapify the affected
            // sub-tree
            heapify(arr, N, largest);
        }
    }
    template<class T>
    // Main function to do heap sort
     void heapSort(T arr[], int N)
    {

        // Build max heap
        for (int i = N / 2 - 1; i >= 0; i--)

            heapify(arr, N, i);

        // Heap sort
        for (int i = N - 1; i >= 0; i--) {

            swap(&arr[0], &arr[i]);

            // Heapify root element to get highest element at
            // root again
            heapify(arr, i, 0);
        }
    }




    template<class T>
    // A function to implement bubble sort
     void bubbleSort(T arr[], int n)
    {
        int i, j;
        for (i = 0; i < n - 1; i++)

            // Last i elements are already in place
            for (j = 0; j < n - i - 1; j++)
                if (arr[j] > arr[j + 1])
                    swap(&arr[j], &arr[j + 1]);
    }


    template<class T>
      void selectionSort(T arr[], int n)
    {
        int i, j, min_idx;
        //printf("Sorting\n   %d ", n);

        // One by one move boundary of unsorted subarray
        for (i = 0; i < n - 1; i++)
        {
            // Find the minimum element in unsorted array
            min_idx = i;
            for (j = i + 1; j < n; j++)
                if (arr[j] < arr[min_idx])
                    min_idx = j;

            // Swap the found minimum element with the first element
            if (min_idx != i)
                swap(&arr[min_idx], &arr[i]);
        }
    }
}

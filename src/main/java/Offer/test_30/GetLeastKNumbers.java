package Offer.test_30;

import java.util.*;

//得到最小的K个数
public class GetLeastKNumbers {
    public static void main(String[] args) {

    }

    /**
     * 思路一：使用快速排序的思想
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] nums, int k) {
        ArrayList<Integer> ret = new ArrayList<>();
        if (k > nums.length || k <= 0)
            return ret;
        findKthSmallest(nums, k - 1);
        /* findKthSmallest 会改变数组，使得前 k 个数都是最小的 k 个数 */
        for (int i = 0; i < k; i++)
            ret.add(nums[i]);
        return ret;
    }

    public void findKthSmallest(int[] nums, int k) {
        int l = 0, h = nums.length - 1;
        while (l < h) {
            int j = partition(nums, l, h);
            if (j == k)
                break;
            if (j > k)
                h = j - 1;
            else
                l = j + 1;
        }
    }

    private int partition(int[] nums, int l, int h) {
        int p = nums[l];     /* 切分元素 */
        int i = l, j = h + 1;
        while (true) {
            while (i != h && nums[++i] < p) ;
            while (j != l && nums[--j] > p) ;
            if (i >= j)
                break;
            swap(nums, i, j);
        }
        swap(nums, l, j);
        return j;
    }

    private void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }

    /**
     * 使用堆
     */
    public ArrayList<Integer> GetLeastNumbers(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        int length = input.length;
        if(k > length || k == 0){
            return result;
        }
        PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(k, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }
        });
        for (int i = 0; i < length; i++) {
            if (maxHeap.size() != k) {
                maxHeap.offer(input[i]);
            } else if (maxHeap.peek() > input[i]) {
                Integer temp = maxHeap.poll();
                temp = null;
                maxHeap.offer(input[i]);
            }
        }
        for (Integer integer : maxHeap) {
            result.add(integer);
        }
        return result;
    }

    /**
     * 使用红黑树。
     */
    public ArrayList<Integer> GetLeastKNumbers(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<>();
        TreeSet<Integer> treeset = new TreeSet<>();
        treeset.clear();
        if(input == null || input.length == 0)
            return null;
        if (k < 1 || input.length < k)
            return result;
        for (int i = 0; i < input.length; i++) {
            if (treeset.size() < k)
                treeset.add(input[i]);
            else {
                int a = treeset.last();
                if (input[i] < a) {
                    treeset.remove(a);
                    treeset.add(input[i]);
                }
            }
        }
        Iterator<Integer> it = treeset.iterator();
        while (it.hasNext()) {
            result.add(it.next());
        }
        return result;
    }
}

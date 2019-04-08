package Offer.test_65;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.PriorityQueue;

public class MaxInWindow {
    public static void main(String[] args) {

    }

    /**
     * 使用大顶堆
     * @param num
     * @param size
     * @return
     */
//    public ArrayList<Integer> maxInWindows(int[] num, int size) {
//        ArrayList<Integer> ret = new ArrayList<>();
//        if (size > num.length || size < 1)
//            return ret;
//        PriorityQueue<Integer> heap = new PriorityQueue<>((o1, o2) -> o2 - o1);  /* 大顶堆 */
//        for (int i = 0; i < size; i++)
//            heap.add(num[i]);
//        ret.add(heap.peek());
//        for (int i = 1, j = i + size - 1; j < num.length; i++, j++) {            /* 维护一个大小为 size 的大顶堆 */
//            heap.remove(num[i - 1]);
//            heap.add(num[j]);
//            ret.add(heap.peek());
//        }
//        return ret;
//    }

    /**
     * 使用两个栈模拟队列
     */
    public ArrayList<Integer> maxInWindows(int [] num, int size) {
        ArrayList<Integer> arr = new ArrayList<>();
        if (num == null)
            return arr;

        if (num.length < size || size <= 0)
            return arr;
        Deque<Integer> queue = new LinkedList<>();
        for (int i=0; i<num.length; i++){
            while (!queue.isEmpty() && num[i] >= num[queue.getLast()])
                queue.pollLast();
            while (!queue.isEmpty() && queue.getFirst() < i - (size-1))
                queue.pollFirst();
            queue.offerLast(i);
            if (i + 1 >= size)
                arr.add(num[queue.getFirst()]);
        }
        return arr;

    }

}

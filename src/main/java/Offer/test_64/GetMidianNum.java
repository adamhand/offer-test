package Offer.test_64;


import java.util.Comparator;
import java.util.PriorityQueue;

//得到数据流中的中位数
public class GetMidianNum {
    //右边是小顶堆
    private static PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    //左边是大顶堆，右半边元素全部大于左半边元素。
    private static PriorityQueue<Integer> maxHeap = new PriorityQueue<>(15, new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            return o2 - o1;
        }
    });
    //当前数据流中读入的元素个数
    private static int count = 0;

    public static void insert(int value){
        if((count & 0x01) == 0){
            /* count 为偶数的情况下插入到右半边。
             * 因为右半边元素都要大于左半边，但是新插入的元素不一定比左半边元素来的大，
             * 因此需要先将元素插入左半边，然后利用左半边为大顶堆的特点，取出堆顶元素即为最大元素，此时插入右半边 */
            maxHeap.add(value);
            minHeap.add(maxHeap.poll());
        }else {
            minHeap.add(value);
            maxHeap.add(minHeap.poll());
        }
        count++;
    }

    public static double getMidian(){
        if((count & 0x01) == 0)
            return (minHeap.peek() + maxHeap.peek()) / 2.0; //偶数个，返回平均数
        else
            return (double)minHeap.peek(); //奇数个，返回小顶堆堆顶元素。因为一开始先插入的是小顶堆，小顶堆多一个元素
    }

    /**
     * 测试
     */
    public static void main(String[] args) {
//        int[] nums = {1,2,3,4,5,6,7,8,9, 10};
//        int[] nums = {1};
        int[] nums = null;

        for(int i = 0; i < nums.length; i++){
            insert(nums[i]);
        }
        System.out.println(getMidian());
    }
}

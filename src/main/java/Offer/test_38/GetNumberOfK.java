package Offer.test_38;

//数组在排序数组中出现的次数
public class GetNumberOfK {
    public static void main(String[] args) {
        int[] nums = {1,2,3,3,3,3,4,5};
        int[] num = {1};
        System.out.println(getNumberOfK(nums, 1));
//        System.out.println(getNumberOfK(null, 6));
//        System.out.println(getNumberOfK(num, 1));
    }

    /**
     * 两次二分查找。递归
     */
//    private static int getFirstK(int[] nums, int k, int start, int end){
//        if(start > end)
//            return -1;
//        int mid = (start + end) >> 1;
//        int midValue = nums[mid];
//
//        if(midValue == k){
//            if((mid > 0 && nums[mid - 1] != k) || mid == 0)
//                return mid;
//            else
//                end = mid - 1;
//        }else if(midValue > k)
//            end = mid - 1;
//        else
//            start = mid + 1;
//
//        return getFirstK(nums, k, start, end);
//    }
//    private static int getLastK(int[] nums, int k, int start, int end){
//        if(start > end)
//            return -1;
//        int mid = (start + end) >> 1;
//        int midValue = nums[mid];
//
//        if(midValue == k){
//            if((mid < nums.length - 1 && nums[mid + 1] != k) || mid == nums.length - 1)
//                return mid;
//            else
//                start = mid + 1;
//        }else if(midValue < k)
//            start = mid + 1;
//        else
//            end = mid - 1;
//        return getLastK(nums, k, start, end);
//    }
//
//    public static int getNumberOfK(int[] nums, int k){
//        if(nums == null || nums.length == 0)
//            return -1;
//        int number = 0;
//        int first = getFirstK(nums, k, 0, nums.length - 1);
//        int last = getLastK(nums, k, 0, nums.length - 1);
//        if(first > -1 && last > -1)
//            number = last - first + 1;
//        return number;
//    }

    /**
     * 简化写法
     */
    public static int getNumberOfK(int[] nums, int K) {
        int first = binarySearch(nums, K);
        int last = binarySearch(nums, K + 1);  //能找到最右边的K的右边的元素
        //return (first == nums.length || first == -1 || nums[first] != K) ? 0 : last - first;
        int number = 0;
        if(first > -1 && last > -1)
            number =  last - first;
        return number;
    }

    private static int binarySearch(int[] nums, int K) {
        if(nums == null || nums.length == 0)
            return -1;
        int l = 0, h = nums.length;
        while (l < h) {
            int m = l + (h - l) / 2;
            //最终能找到最左边的K;如果不存在K，会找到比K小的一个数
            if (nums[m] >= K)
                h = m;
            else
                l = m + 1;
        }
        return l;
    }
}

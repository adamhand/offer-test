package Offer.test_41;

//和为S的两个数字
public class FindNumbersWithSum {
    public static void main(String[] args) {
        int[] nums = {1,2,4,7,11,15};
        int[] ans = new int[2];
        ans = findNumberWithSum(nums, 15);
        ans = findNumberWithSum(null, 11);
        System.out.println(ans[0] +" " + ans[1]);
    }

    /**
     * 双指针法
     * @param nums
     * @param sum
     * @return
     */
    public static int[] findNumberWithSum(int[] nums, int sum){
        if(nums == null || nums.length < 2)
            throw new IllegalArgumentException("nums size is not correct");
        int head = 0, tail = nums.length - 1;
        while(head < tail){
            if(nums[head] + nums[tail] == sum)
                return new int[]{nums[head], nums[tail]};
            else if(nums[head] + nums[tail] < sum)
                head++;
            else
                tail--;
        }
        return null;
    }
}

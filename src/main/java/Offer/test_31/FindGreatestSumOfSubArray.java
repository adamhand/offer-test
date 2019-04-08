package Offer.test_31;

import java.lang.annotation.Target;

//连续子数组的最大和
public class FindGreatestSumOfSubArray {
    public static void main(String[] args) {
        int[] nums = {1,-2,3,10,-4,7,2,-5};
//        System.out.println(greatestSumOfSubArray(nums));
        System.out.println(greatestSum(nums));
    }


    /**
     * 方法1
     * @param nums
     * @return
     */
    public static int greatestSumOfSubArray(int[] nums){
        if(nums == null || nums.length <= 0){
            System.out.println("Invalid input");
            return -1;
        }
        int curSum = 0;
        //0x80000000 = -2147483648
        int greatestSum = 0x80000000;
        for(int i = 0; i < nums.length; i++){
            if(curSum <= 0)
                curSum = nums[i];
            else
                curSum += nums[i];
            if(curSum > greatestSum)
                greatestSum = curSum;
        }
        return greatestSum;
    }
    /**
     * 方法2：递归法
     */
    public static int greatestSum(int[] nums){
        return greatestSumSub(nums, 0, nums.length - 1);
    }

    private static int greatestSumSub(int[] nums, int left, int right){
        if(nums == null || nums.length < 0)
            return -1;
        if(left == right)
            return Math.max(0, nums[left]);

        int maxLeftSum = 0, maxRightSum = 0; //左右边不包含中间的最大和
        int maxLeftBorderSum = 0, maxRightBorderSum = 0; //包含中间边界的左右边最大和
        int curLeftBorderSum = 0, curRightBorderSum = 0; //包含中间边界的左右边当前和

        //求含中间边界的左右部分的最大值
        int mid = (left + right) >> 1;
        for(int i = mid; i >= left; i--){
            curLeftBorderSum += nums[i];
            maxLeftBorderSum = Math.max(curLeftBorderSum, maxLeftBorderSum);
        }
        for(int i = mid + 1; i <= right; i++){
            curRightBorderSum += nums[i];
            maxRightBorderSum = Math.max(curRightBorderSum, maxRightBorderSum);
        }

        //递归求左右部分最大值
        maxLeftSum = greatestSumSub(nums, left, mid);
        maxRightSum = greatestSumSub(nums, mid + 1, right);

        //返回三者中的最大值
        return maxOfThree(maxLeftSum, maxRightSum, maxLeftBorderSum + maxRightBorderSum);
    }
    private static int maxOfThree(int a, int b, int c){
        int max = a;
        if(b > max)
            max = b;
        if(c > max)
            max = c;
        return max;
    }
}

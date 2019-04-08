package Offer.test_44;

import java.util.Arrays;

//扑克牌是不是连续的
public class IsContinuous {
    public static void main(String[] args) {
        int[] nums_1 = {1,1,2,3,4,5};
        int[] nums_2 = {1,2,3,4,5};
        int[] nums_3 = {0,0,1,2,6};
        int[] nums_4 = {0,1,2,4,5};
        int[] nums_5 = {0,1,2,3,5};

        System.out.println(isContinuous(null));
        System.out.println(isContinuous(nums_1));
        System.out.println(isContinuous(nums_2));
        System.out.println(isContinuous(nums_3));
        System.out.println(isContinuous(nums_4));
        System.out.println(isContinuous(nums_5));
    }

    /**
     * 判断扑克牌是不是顺子
     */
    public static boolean isContinuous(int[] nums){
        if(nums == null || nums.length < 1)
            return false;

        Arrays.sort(nums);

        int numberOfZero = 0;
        int numberOfGap = 0;
        //统计0的个数
        for(int i = 0; i < nums.length && nums[i] == 0; i++)
            numberOfZero++;

        //统计间隔的个数
        int left = numberOfZero; //跳过前面的0
        int right = left + 1;
        while(right < nums.length){
            if(nums[left] == nums[right])
                return false;
            numberOfGap += nums[right] - nums[left] - 1;
            left = right;
            right++;
        }
        return (numberOfGap > numberOfZero) ? false : true;
    }
}

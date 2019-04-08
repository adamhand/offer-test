package Offer.test_40;

import com.sun.glass.ui.Size;

import java.util.HashMap;

//数组中只出现一次的数字
public class FindNumberAppendOnce {
    public static void main(String[] args) {
        int[] nums = {2,4,3,6,3,2,5,5};
        int[] ans = new int[2];
//        ans = findNumberAppendOnce(nums);
        System.out.println(ans[0] +" " + ans[1]);
    }

    /**
     * 使用HashMap,但是空间复杂度不是O(1)。
     * @param nums
     * @return
     */
//    public static int[] findNumberAppendOnce(int[] nums){
//        if(nums == null || nums.length == 0)
//            return null;
//        HashMap<Integer, Integer> map = new HashMap<>();
//        int[] result = new int[2];
//        for(int i = 0; i < nums.length; i++){
//            if(map.containsKey(nums[i]))
//                map.put(nums[i], map.get(nums[i]) + 1);
//            else
//                map.put(nums[i], 1);
//        }
//        int i = 0;
//        for(Integer key : map.keySet()){
//            if(map.get(key) == 1)
//                result[i++] = key;
//            if(i >= 2)   //题目中说是有两个，但还是判断一下，避免有错误输出
//                break;
//        }
//        return result;
//    }

    /**
     * 使用异或法
     */
//    public static int[] findNumberAppendOnce(int[] nums){
//        if(nums == null || nums.length < 2)
//            throw new IllegalArgumentException("nums size must bigger than 2");
//
//        int resultExclusiveOR = 0;       //数组中所有值异或一次的结果
//        for(int i = 0; i < nums.length; i++)
//            resultExclusiveOR ^= nums[i];
//        int indexOf1 = findFirstBitIs1(resultExclusiveOR); //最右边的1的位数
//
//        int num1 = 0, num2 = 0;
//        for(int i = 0; i < nums.length; i++){
//            if(isBit1(nums[i], indexOf1))
//                num1 ^= nums[i];
//            else
//                num2 ^= nums[i];
//        }
//        return new int[] {num1, num2};
//    }
//
//    //找到整数num的二进制表示中最右边是1的位
//    private static int findFirstBitIs1(int num){
//        int indexBit = 0;
//        //8 * Integer.SIZE是整型数转化为二进制后最多有多少位
//        while((num & 0x01) == 0 && (indexBit < 8 * Integer.SIZE)){
//            num >>= 1;
//            indexBit++;
//        }
//        return indexBit;
//    }
//    //判断整数num的二进制表示中从右边起的indexBit位是不是1
//    private static boolean isBit1(int num, int indexBit){
//        num >>= indexBit;
//        return (num & 0x01) == 1;
//    }

    /**
     * 异或法简写。diff &= -diff 得到 diff 最右侧不为 0 的位，也就是不存在重复的两个元素在位级表示上最右侧不同的那一位
     */
    public void findNumsAppearOnce(int[] nums, int num1[], int num2[]) {
        int diff = 0;
        for (int num : nums)
            diff ^= num;
        diff &= -diff;
        for (int num : nums) {
            if ((num & diff) == 0)
                num1[0] ^= num;
            else
                num2[0] ^= num;
        }
    }
}

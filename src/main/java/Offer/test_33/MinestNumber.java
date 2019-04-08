package Offer.test_33;

import java.util.Arrays;
import java.util.Comparator;

//将数组排成最小的数
public class MinestNumber {
    public static String minNumber(int[] nums){
        if(nums == null || nums.length == 0)
            return "";
        String[] strs = new String[nums.length];
        for(int i = 0; i < nums.length; i++){
            strs[i] = String.valueOf(nums[i]);
        }
        //重写compareTo方法
        Arrays.sort(strs, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return (o1 + o2).compareTo(o2 + o1);
            }
        });
        StringBuilder sb = new StringBuilder();
        for(String str : strs)
            sb.append(str);

        return sb.toString();
    }

    public static void main(String[] args) {
//        int[] nums = {3,32,321};
//        int[] nums = null;
//        int[] nums = {1};
        int[] nums = {222,111,333,1112};
        System.out.println(minNumber(nums));
    }
}

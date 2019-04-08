package Offer.test_null3;

import com.sun.org.apache.bcel.internal.generic.RETURN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//最长不含重复字符的子字符串
public class LengthOfLongestSubstring {
    public static void main(String[] args) {
        String str = "arabcacfr";
        System.out.println(longestSubstringWithoutDuplication(str));
        System.out.println(longestSubstringWithoutDuplication(""));

        System.out.println(longestSubString(str));
        System.out.println(longestSubString(""));
    }

    /**
     * 使用额外空间
     */
    public static int longestSubstringWithoutDuplication(String str){
        if(str == null || str.length() == 0)
            return -1;
        int maxLength = 1;
        List<Character> list = new ArrayList<>();
        list.add(str.charAt(0));
        for(int i = 0; i < str.length(); i++){
            if(list.contains(str.charAt(i))){
                int index = list.indexOf(str.charAt(i));
                list = list.subList(index + 1, list.size());
                list.add(str.charAt(i));
                maxLength = Math.max(maxLength, list.size());
            }else {
                list.add(str.charAt(i));
                //maxLength++;
                maxLength = Math.max(maxLength, list.size());
            }
        }
        return maxLength;
    }

    /**
     * 使用动态规划
     */
    public static int longestSubString(String str){
        if(str == null || str.length() == 0)
            return -1;
        int curLength = 0, maxLength = 0;
        int[] position = new int[26];
        //初始化为-1，负数表示没出现过
        Arrays.fill(position, -1);
        for(int i = 0; i < str.length(); i++){
            int c = str.charAt(i) - 'a';
            int preIndex = position[c];
            //当前字符第一次出现，或者前一个非重复子字符串中没有包含当前字符
            if(preIndex == -1 || i - preIndex > curLength){
                curLength++;
            }else {
                //更新最长非重复子字符串的长度
                if(maxLength < curLength)
                    maxLength = curLength;
                curLength = i - preIndex;
            }
            position[c] = i;  //更新字符出现的位置
        }
        maxLength = Math.max(maxLength, curLength);
        return maxLength;
    }
}

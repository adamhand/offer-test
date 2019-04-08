package Offer.test_35;

import java.util.BitSet;
import java.util.HashMap;

//第一个非重复的字符
public class FirstNotRepeatingChar {
    public static void main(String[] args) {
        String str = "abaccdeff";
        System.out.println(firstNotRepeatChar(str));
        System.out.println(firstNotRepeatChar("abcdef"));
        System.out.println(firstNotRepeatChar("aabbccééééééé"));
        System.out.println(firstNotRepeatingChar("aabbccééééééé"));
        System.out.println(firstNotRepeatingChar("éasdffhtgh")); //é是ASCII码表中后128位的数，主要是为了验证后128位的编码是整数，要不然数组会越界
    }

    /**
     * 使用HashMap
     */
    public static char firstNotRepeatChar(String str){
        if(str == null || str.length() == 0)
            return '\0';

        HashMap<Character, Integer> map = new HashMap<>();
        for(int i = 0; i < str.length(); i++){
            if(map.containsKey(str.charAt(i)))
                map.put(str.charAt(i), map.get(str.charAt(i)) + 1);
            else
                map.put(str.charAt(i), 1);
        }
        for(int i = 0; i < str.length(); i++){
            if(map.get(str.charAt(i)) == 1)
                return str.charAt(i);
        }
        return '\0';   //没找到，返回一个空字符
    }

    /**
     * 使用数组进行扩展
     */
//    public static char firstNotRepeatingChar(String str) {
//            if(str == null || str.length() == 0)
//            return '\0';
//        int[] cnts = new int[256];
//        for (int i = 0; i < str.length(); i++)
//            cnts[str.charAt(i)]++;
//        for (int i = 0; i < str.length(); i++)
//            if (cnts[str.charAt(i)] == 1)
//                return str.charAt(i);
//        return '\0';
//    }

    /**
     * 使用位集
     */
    public static char firstNotRepeatingChar(String str) {
        if(str == null || str.length() == 0)
            return '\0';

        BitSet bs1 = new BitSet(256);
        BitSet bs2 = new BitSet(256);
        for (char c : str.toCharArray()) {
            //如果两个位集都没有，就放到bs1中
            if (!bs1.get(c) && !bs2.get(c))
                bs1.set(c);     // 0 0 -> 0 1
            //如果只有bs1中有，bs2中也会放一个。经过这两步，只出现一次的字符放在bs1中，出现多次的字符bs1和bs2中都有
            else if (bs1.get(c) && !bs2.get(c))
                bs2.set(c);     // 0 1 -> 1 1
        }
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (bs1.get(c) && !bs2.get(c))  // 0 1
                return c;
        }
        return '\0';
    }
}

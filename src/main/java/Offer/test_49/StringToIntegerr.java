package Offer.test_49;

public class StringToIntegerr {
    public static void main(String[] args) {
        //直接用测试用例来测试
        //正常情况下的3中不同的输入
        String str1="123";
        String str2="-123";
        String str3="+123";
        //只输入一个"+"  或者是  "-"
        String str4="+" ;
        String str5="-";
        //不可以转换的输入
        String str6="abc123";
        String str7="";
        String str8=null;

        System.out.println(strToInt(str1));
        System.out.println(strToInt(str2));
        System.out.println(strToInt(str3));
        System.out.println(strToInt(str4));
        System.out.println(strToInt(str5));
        System.out.println(strToInt(str6));
        System.out.println(strToInt(str7));
        System.out.println(strToInt(str8));
    }

    public static int strToInt(String str){
        if (str == null || str.length() == 0)
            return 0;
        boolean isNegative = str.charAt(0) == '-';
        int ret = 0;
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (i == 0 && (c == '+' || c == '-'))  /* 符号判定 */
                continue;
            if (c < '0' || c > '9')                /* 非法输入 */
                return 0;
            ret = ret * 10 + (c - '0');
        }
        return isNegative ? -ret : ret;
    }
}
